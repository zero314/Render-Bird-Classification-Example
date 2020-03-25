import aiohttp
import asyncio
import uvicorn
from fastai import *
from fastai.vision import *
from io import BytesIO
from starlette.applications import Starlette
from starlette.middleware.cors import CORSMiddleware
from starlette.responses import HTMLResponse, JSONResponse
from starlette.staticfiles import StaticFiles

export_file_url = 'https://drive.google.com/file/d/1WMOOWAve4V60LQf4gkaGG1QqVxrNX6L0/view?usp=sharing'
export_file_name = 'export.pkl'

classes = ['ALBATROSS',
  'ALEXANDRINE PARAKEET',
  'AMERICAN GOLDFINCH',
  'AMERICAN KESTREL',
  'AMERICAN REDSTART',
  'ANNAS HUMMINGBIRD',
  'BALD EAGLE',
  'BALTIMORE ORIOLE',
  'BANANAQUIT',
  'BAR-TAILED GODWIT',
  'BARN OWL',
  'BARN SWALLOW',
  'BAY-BREASTED WARBLER',
  'BELTED KINGFISHER',
  'BIRD OF PARADISE',
  'BLACK SKIMMER',
  'BLACK-CAPPED CHICKADEE',
  'BLACK-NECKED GREBE',
  'BLACKBURNIAM WARBLER',
  'BLUE HERON',
  'BOBOLINK',
  'BROWN THRASHER',
  'CACTUS WREN',
  'CALIFORNIA CONDOR',
  'CALIFORNIA GULL',
  'CALIFORNIA QUAIL',
  'CAPE MAY WARBLER',
  'CHARA DE COLLAR',
  'CHIPPING SPARROW',
  'CINNAMON TEAL',
  'COCK OF THE  ROCK',
  'COCKATOO',
  'COMMON LOON',
  'COUCHS KINGBIRD',
  'CRESTED AUKLET',
  'CRESTED CARACARA',
  'CROW',
  'CROWNED PIGEON',
  'CURL CRESTED ARACURI',
  'DARK EYED JUNCO',
  'DOWNY WOODPECKER',
  'EASTERN BLUEBIRD',
  'EASTERN ROSELLA',
  'EASTERN TOWEE',
  'ELEGANT TROGON',
  'EMPEROR PENGUIN',
  'EVENING GROSBEAK',
  'FLAME TANAGER',
  'FLAMINGO',
  'FRIGATE',
  'GOLD WING WARBLER',
  'GOLDEN CHLOROPHONIA',
  'GOLDEN EAGLE',
  'GOLDEN PHEASANT',
  'GOULDIAN FINCH',
  'GRAY CATBIRD',
  'GREY PLOVER',
  'HAWAIIAN GOOSE',
  'HOODED MERGANSER',
  'HOOPOES',
  'HOUSE FINCH',
  'HOUSE SPARROW',
  'HYACINTH MACAW',
  'INDIGO BUNTING',
  'LARK BUNTING',
  'LILAC ROLLER',
  'LONG-EARED OWL',
  'MALLARD DUCK',
  'MANDRIN DUCK',
  'MARABOU STORK',
  'MOURNING DOVE',
  'MYNA',
  'NICOBAR PIGEON',
  'NORTHERN CARDINAL',
  'NORTHERN FLICKER',
  'NORTHERN GOSHAWK',
  'NORTHERN MOCKINGBIRD',
  'OSTRICH',
  'PAINTED BUNTIG',
  'PARADISE TANAGER',
  'PARUS MAJOR',
  'PEACOCK',
  'PELICAN',
  'PEREGRINE FALCON',
  'PINK ROBIN',
  'PUFFIN',
  'PURPLE FINCH',
  'PURPLE GALLINULE',
  'PURPLE MARTIN',
  'QUETZAL',
  'RAINBOW LORIKEET',
  'RED FACED CORMORANT',
  'RED HEADED WOODPECKER',
  'RED THROATED BEE EATER',
  'RED WINGED BLACKBIRD',
  'RED WISKERED BULBUL',
  'RING-NECKED PHEASANT',
  'ROADRUNNER',
  'ROBIN',
  'ROUGH LEG BUZZARD',
  'RUBY THROATED HUMMINGBIRD',
  'SAND MARTIN',
  'SCARLET IBIS',
  'SCARLET MACAW',
  'SNOWY EGRET',
  'SPLENDID WREN',
  'STORK BILLED KINGFISHER',
  'STRAWBERRY FINCH',
  'TEAL DUCK',
  'TIT MOUSE',
  'TOUCHAN',
  'TRUMPTER SWAN',
  'TURKEY VULTURE',
  'TURQUOISE MOTMOT',
  'VENEZUELIAN TROUPIAL',
  'VERMILION FLYCATHER',
  'WESTERN MEADOWLARK',
  'WILSONS BIRD OF PARADISE',
  'WOOD DUCK',
  'YELLOW HEADED BLACKBIRD']
path = Path(__file__).parent

app = Starlette()
app.add_middleware(CORSMiddleware, allow_origins=['*'], allow_headers=['X-Requested-With', 'Content-Type'])
app.mount('/static', StaticFiles(directory='app/static'))


async def download_file(url, dest):
    if dest.exists(): return
    async with aiohttp.ClientSession() as session:
        async with session.get(url) as response:
            data = await response.read()
            with open(dest, 'wb') as f:
                f.write(data)


async def setup_learner():
    await download_file(export_file_url, path / export_file_name)
    try:
        learn = load_learner(path, export_file_name)
        return learn
    except RuntimeError as e:
        if len(e.args) > 0 and 'CPU-only machine' in e.args[0]:
            print(e)
            message = "\n\nThis model was trained with an old version of fastai and will not work in a CPU environment.\n\nPlease update the fastai library in your training environment and export your model again.\n\nSee instructions for 'Returning to work' at https://course.fast.ai."
            raise RuntimeError(message)
        else:
            raise


loop = asyncio.get_event_loop()
tasks = [asyncio.ensure_future(setup_learner())]
learn = loop.run_until_complete(asyncio.gather(*tasks))[0]
loop.close()


@app.route('/')
async def homepage(request):
    html_file = path / 'view' / 'index.html'
    return HTMLResponse(html_file.open().read())


@app.route('/analyze', methods=['POST'])
async def analyze(request):
    img_data = await request.form()
    img_bytes = await (img_data['file'].read())
    img = open_image(BytesIO(img_bytes))
    prediction = learn.predict(img)[0]
    return JSONResponse({'result': str(prediction)})


if __name__ == '__main__':
    if 'serve' in sys.argv:
        uvicorn.run(app=app, host='0.0.0.0', port=5000, log_level="info")
