from django.shortcuts import render
from django.http import HttpResponse
from .forms import ImageUploadForm
from PIL import Image
import torch
import torchvision.transforms as transforms
import torchvision.models as models
import json


def index(request):
    if request.method == 'POST':
        form = ImageUploadForm(request.POST, request.FILES)
        if form.is_valid():
            image_file = form.cleaned_data['image_upload']
            image = Image.open(image_file)

            # Преобразование изображения для классификации
            transform = transforms.Compose([
                transforms.Resize(256),
                transforms.CenterCrop(224),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])
            ])
            image_tensor = transform(image)
            image_tensor = image_tensor.unsqueeze(0)

            # Загрузка предварительно обученной модели ResNet18
            model = models.resnet18(pretrained=True)
            model.eval()

            # Классификация изображения
            output = model(image_tensor)
            predictions = torch.argmax(output, dim=1)

            # Отправка результатов классификации на страницу result
            category_mapping = json.loads(open(
                'classifier/category_mapping.json').read())
            prediction_label = category_mapping[str(predictions.item())]
            return render(
                request, 'result.html', {'result': prediction_label})
        else:
            form = ImageUploadForm()
        return render(request, 'index.html', {'form': form})
