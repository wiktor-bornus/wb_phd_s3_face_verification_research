from facenet_pytorch import InceptionResnetV1, MTCNN
from torchvision.transforms.functional import to_pil_image, pil_to_tensor
import torch
import matplotlib.pyplot as plt
from PIL import Image


def extract_face_embedding(resnet, path, img=None, tolist=True, mtcnn=None):
    # try:
    img = Image.open(path)
    print(img.size)
    face = mtcnn(img)

    if face is None:
        print(f"No face detected: {path}")
        return None

    face = face.unsqueeze(0).to('cuda')

    with torch.no_grad():
        embedding = resnet(face)

    embedding = embedding / embedding.norm(dim=1, keepdim=True)

    if tolist:
        return embedding.squeeze().cpu().tolist()
    else:
        return embedding.squeeze()


if __name__ == "__main__":
    resnet = InceptionResnetV1(pretrained="vggface2", device=torch.device("cuda")).eval()
    mtcnn = MTCNN(image_size=160, thresholds=[0.9, 0.9, 0.9], device=torch.device("cuda"))

    face_1_path = "/home/wiktor/Doktorat/repositories/mavceleb_baseline/dataset/mavceleb_v1_train/faces/id0001/English/HpzYIwAaNlE/0002300.jpg"
    face_2_path = "/home/wiktor/Doktorat/repositories/mavceleb_baseline/dataset/mavceleb_v1_train/faces/id0001/English/HpzYIwAaNlE/0001225.jpg"

    emb1 = extract_face_embedding(resnet, face_1_path, tolist=False, mtcnn=mtcnn)
    emb2 = extract_face_embedding(resnet, face_1_path, tolist=False, mtcnn=mtcnn)

    print(emb1)
    print(emb2)

