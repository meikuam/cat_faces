import sys
sys.path.append('.')
import logging


from src.stonks_bot.stonks_bot import StonksBot
from src.model.model_image_processor import ModelWorker, ModelImageProcessor
from src.sticker.sticker_image_processor import StickerImageProcessor
from src.image_processor.image_processor import ComposeImageProcessor
from src.text_processor.dumb_text_processor import DumbTextProcessor


if __name__ == "__main__":
    logging.basicConfig(filename='messages.log', level=logging.INFO)
    bot_token = ''
    traced_model_path = 'traced_model.pt'

    dump_messages = True
    compose_image_processor = ComposeImageProcessor([
        ModelWorker(
            traced_model_path=traced_model_path,
            image_shape=(416, 416)
        ),
        StickerImageProcessor(
            erode_thickness=3,
            edge_thickness=5,
            edge_color=(150, 150, 150, 255)
        )
    ])
    dumb_text_processor = DumbTextProcessor(dump_messages)
    print("start_bot")
    stonks_bot = StonksBot(
        token=bot_token,
        dump_messages=dump_messages,
        image_processor=compose_image_processor,
        text_processor=dumb_text_processor
    )
    print("bot started")
