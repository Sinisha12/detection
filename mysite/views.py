from django.shortcuts import render
from django.http.response import HttpResponse
import funk_slika



def landing_page(request):
    if request.method == 'GET':
        return render(request, 'landing.html')
    if request.method == 'POST':
        # TODO logic goes here
        my_image = request.FILES['myimage']
        funk_slika.slika(my_image)

        image_data = open('/home/sinisha/venv/mysite/image_return2.jpg', mode='rb').read()
        return HttpResponse(image_data, content_type="image/jpg")
