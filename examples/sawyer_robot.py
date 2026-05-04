from fastapi import FastAPI, UploadFile, File, Form
from fastapi.responses import JSONResponse
import numpy as np
import cv2
import io
from openpi_client.websocket_client_policy import WebsocketClientPolicy
from typing import Optional
from PIL import Image
import threading

# ----------------------------------------------------------------------
# 1. 初始化 FastAPI 服务
# ----------------------------------------------------------------------
app: FastAPI = FastAPI(
    title="OpenPI Robot Action Server",
    version="0.1.0",
    description="Receives images and robot state, calls OpenPI policy server via WebSocket to predict actions."
)


# ----------------------------------------------------------------------
# 2. 启动时连接策略服务器
# ----------------------------------------------------------------------
POLICY_HOST = "127.0.0.1"
POLICY_PORT = 8000
policy_client: Optional[WebsocketClientPolicy] = None


@app.on_event("startup")
def startup_event():
    global policy_client
    def _init_client():
        global policy_client
        try:
            print("[OpenPI-Server] Connecting to policy server...")
            policy_client = WebsocketClientPolicy(host=POLICY_HOST, port=POLICY_PORT)
            print("[OpenPI-Server] Connected ✔ Metadata:", policy_client.get_server_metadata())
        except Exception as e:
            print("[OpenPI-Server] Failed to connect:", e)
    threading.Thread(target=_init_client, daemon=True).start()

# ----------------------------------------------------------------------
# 3. FastAPI 预测接口
# ----------------------------------------------------------------------
@app.post("/predict")
async def predict(
    image: UploadFile = File(..., description="Main RGB image from camera"),
    wrist_image: Optional[UploadFile] = File(None, description="Optional wrist camera image"),
    state: str = Form(..., description="Comma-separated 7 joint values, e.g. '0.1,0.2,...'"),
    task: str = Form("pick up the cube", description="Instruction or prompt for the model")
):
    """
    Receive an RGB frame + optional wrist image + robot state,
    send to OpenPI WebSocket policy server, and return the predicted action.
    """
    global policy_client

    if policy_client is None:
        return JSONResponse({"error": "Policy client not connected yet."}, status_code=503)

    # Decode main image
    main_bytes = await image.read()
    main_img = Image.open(io.BytesIO(main_bytes)).convert("RGB")
    main_img = np.array(main_img)
    main_img = cv2.resize(main_img, (224, 224))
    print(wrist_image)
    # Decode wrist image or fallback to main image
    if wrist_image is not None:
        wrist_bytes = await wrist_image.read()
        wrist_img = Image.open(io.BytesIO(wrist_bytes)).convert("RGB")
        wrist_img = np.array(wrist_img)
        wrist_img = cv2.resize(wrist_img, (224, 224))
    else:
        wrist_img = main_img

    # Parse robot state string -> float array
    try:
        joint_values = np.array([float(x.strip()) for x in state.split(",")], dtype=np.float32)
    except ValueError:
        return JSONResponse({"error": f"Invalid 'state' format: {state}"}, status_code=400)

    # Build request dict
    request_data = {
        "observation/image": main_img,
        "observation/wrist_image": wrist_img,
        "observation/state": joint_values,
        "prompt": task,
    }

    # Call the OpenPI inference
    try:
        result = policy_client.infer(request_data)
        actions = result.get("actions", [])
        print("actions: ", np.array(actions).tolist())
        return {"actions": np.array(actions).tolist()}
    except Exception as e:
        return JSONResponse({"error": f"Inference failed: {e}"}, status_code=500)

# ----------------------------------------------------------------------
# 4. 启动入口（开发测试用）
# ----------------------------------------------------------------------
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "sawyer_robot:app",
        host="0.0.0.0",
        port=8001,
        reload=False,
        )
