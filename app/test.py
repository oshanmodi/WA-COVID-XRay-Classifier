from pathlib import Path


model_file_url = 'https://drive.google.com/uc?export=download&id=1r1H3eoDmE-9Looy4sKhrGTHJ4ixdRAmr'
model_file_name = 'covid_classification'
classes = ['COVID-19', 'Viral Pneumonia', 'NORMAL']
path = Path(__file__).parent



async def download_file(url, dest):
    if dest.exists(): return
    async with aiohttp.ClientSession() as session:
        async with session.get(url) as response:
            data = await response.read()
            with open(dest, 'wb') as f: f.write(data)

async def setup_learner():
    await download_file(model_file_url, path/'models'/f'{model_file_name}.pkl')
    data_bunch = ImageDataBunch.single_from_classes(path, classes,
        ds_tfms=get_transforms(), size=512).normalize(imagenet_stats)
    learn = cnn_learner(data_bunch, models.resnet34, pretrained=False)
    learn.load(model_file_name)
    print("completed")
    return learn


setup_learner()