import insightface
import onnxruntime as ort


if __name__ == "__main__":
    model_path = "./model_R100_MS1MV2.onnx"

    providers = (
        ["CUDAExecutionProvider", "CPUExecutionProvider"]
        if "CUDAExecutionProvider" in ort.get_available_providers()
        else ["CPUExecutionProvider"]
    )

    arcface = insightface.model_zoo.get_model(
        model_path,
        providers=providers
    )

    arcface.prepare(ctx_id=0 if "CUDAExecutionProvider" in providers else -1)

    print("Model loaded successfully")