import gradio as gr
from torchvision import transforms
from model import Generator
from utils import *
import config

def draw(img):
    model_path = 'pretrained/gen_best.pth'
    
    transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])
    x = transform(img['composite']).unsqueeze(0).to(config.device)
    gen = Generator().to(config.device)
    load_checkpoint(gen, model_path)
    gen.eval()
    with torch.no_grad():
        y = gen(x)
        y = denormalize(y)
        y = y.squeeze(0)
        print(y.shape)
    return (transforms.ToPILImage())(y)

demo = gr.Interface(
    fn=draw,
    inputs=gr.Sketchpad(type='pil', image_mode='RGB', interactive=True),
    outputs=gr.Image(width=256, height=256, format='png'),
    examples='dataset/edge2face/test',
    title="Edge2face",
    description="See how your hand drawing faces look in real life. Please follow the style of below examples to achieve the best result."
)

demo.launch()