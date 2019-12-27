import os
from threading import Thread, Lock
from telegram import Update
from telegram.ext import Updater, CommandHandler, MessageHandler, Filters, CallbackContext
import time
import logging

import io
import skimage.io as skio
from PIL import Image
import numpy as np
import traceback

from src.image_processor.image_processor import ImageProcessor
from src.text_processor.text_processor import TextProcessor
from src.utils.helpers import imread

class StonksBot:
    
    def __init__(
            self,
            token,
            proxy_url=None,
            dump_messages=False,
            image_processor=ImageProcessor(),
            text_processor=TextProcessor()
    ):
        logging.info(f"{self.__class__.__name__}: create")
        # token from botFather
        self.token = token
        self.dump_messages = dump_messages  # TODO: dump messages to local storage
        try:
            if not os.path.isdir("images"):
                os.mkdir("images")
            self.dump_images_path = "images"
        except Exception:
            self.dump_images_path = "."
        self.image_processor = image_processor
        self.text_processor = text_processor

        request_kwargs = {
            'read_timeout': 6, 'connect_timeout': 7
        }
        # TODO: support proxy
        if proxy_url is not None:
            request_kwargs['proxy_url'] = proxy_url
        self.updater = Updater(
            token=self.token,
            request_kwargs=request_kwargs,
            use_context=True
        )
        self.dispatcher = self.updater.dispatcher

        def startCommand(update: Update, context: CallbackContext):
            # подробнее об объекте update: https://core.telegram.org/bots/api#update
            chat_id = update.message.chat_id

            logging.info(f"{self.__class__.__name__}: started {chat_id}")
            intro_message = """Привет, пришли мне картинку с котиком, а я сделаю из него стикер."""
            context.bot.send_message(chat_id=chat_id, text=intro_message)

        def textMessage(update: Update, context: CallbackContext):
            chat_id = update.message.chat_id
            message = update.message.text
            self.text_processor(chat_id, message, context)

        def photoMessage(update: Update, context: CallbackContext):
            chat_id = update.message.chat_id
            try:
                if isinstance(update.message.photo, list):
                    photo = update.message.photo[-1]
                    image_file_id = photo.file_id
                    logging.info(f"{self.__class__.__name__}: photo message, {chat_id}")

                    image_data = self.get_numpy_image(image_file_id, context)
                    if image_data is not None:
                        if self.dump_messages:
                            self.dump_image(image_data, f"{chat_id}_{time.time()}.png")
                        context.bot.send_message(
                            chat_id=chat_id,
                            text='получил фото'
                        )
                        # processing
                        processed_image_data = self.image_processor(image_data)

                        # send result
                        self.send_numpy_image(
                            image_data=processed_image_data,
                            image_type="sticker",
                            chat_id=chat_id,
                            context=context
                        )
                    else:
                        raise ValueError("error while processing")
                else:
                    raise ValueError("photo message not a list")
            except Exception:
                context.bot.send_message(chat_id=chat_id, text='что-то пошло не так')
                logging.warning(f"{self.__class__.__name__}: photo smth wrong {traceback.format_exc()}")

        def imageMessage(update: Update, context: CallbackContext):
            chat_id = update.message.chat_id
            try:
                image_file_id = update.message.document.file_id
                logging.info(f"{self.__class__.__name__}: image message, {chat_id}")
                image_data = self.get_numpy_image(image_file_id, context)
                if image_data is not None:
                    if self.dump_messages:
                        self.dump_image(image_data, f"{chat_id}_{time.time()}.png")
                    context.bot.send_message(
                        chat_id=chat_id,
                        text='получил изображение'
                    )
                    # processing
                    processed_image_data = self.image_processor(image_data)
                    # send result
                    self.send_numpy_image(
                        image_data=processed_image_data,
                        image_type="document",
                        chat_id=chat_id,
                        context=context
                    )
                else:
                    raise ValueError("image message can't get it")
            except Exception:
                context.bot.send_message(chat_id=chat_id, text='что-то пошло не так')
                logging.warning(f"{self.__class__.__name__}: photo smth wrong {traceback.format_exc()}")

        def randomCommand(update: Update, context: CallbackContext):
            """get image from thiscatdoesnotexist.com, and make sticker from it."""

            chat_id = update.message.chat_id
            try:
                url = 'https://thiscatdoesnotexist.com/'
                image_data = imread(url)
                if self.dump_messages:
                    self.dump_image(image_data, f"{chat_id}_{time.time()}.png")
                # processing
                processed_image_data = self.image_processor(image_data)
                # print(processed_image_data.shape)
                # send result
                self.send_numpy_image(
                    image_data=processed_image_data,
                    image_type="sticker",
                    chat_id=chat_id,
                    context=context
                )
            except Exception:
                context.bot.send_message(chat_id=chat_id, text='что-то пошло не так')
                logging.warning(f"{self.__class__.__name__}: photo smth wrong {traceback.format_exc()}")

        # def documentMessage(update: Update, context: CallbackContext):
        #     chat_id = update.message.chat_id
        #     print('document message, ', chat_id)
        #     context.bot.send_message(chat_id=chat_id, text='получил документ')

        # обработчики
        start_command_handler = CommandHandler('start', startCommand)
        random_command_handler = CommandHandler('random', randomCommand)
        text_message_handler = MessageHandler(Filters.text, textMessage)
        photo_message_handler = MessageHandler(Filters.photo, photoMessage)
        image_message_handler = MessageHandler(Filters.document.image, imageMessage)
        # document_message_handler = MessageHandler(Filters.document, documentMessage)

        # регистрируем обработчики
        self.dispatcher.add_handler(start_command_handler)
        self.dispatcher.add_handler(random_command_handler)
        self.dispatcher.add_handler(text_message_handler)
        self.dispatcher.add_handler(photo_message_handler)
        self.dispatcher.add_handler(image_message_handler)
        # self.dispatcher.add_handler(document_message_handler)

        # запускаем пуллинг сообщений
        # TODO: message polling can fall we should do smth else
        self.updater.start_polling(clean=False)

    def get_numpy_image(self, file_id: str, context: CallbackContext):
        """
        Get numpy image with file_id from context of message

        :param file_id: id of file
        :param context: context of message
        :return: np.ndarray if success else None
        """
        image_data = None
        try:
            file_info = context.bot.get_file(file_id)
            buffered = io.BytesIO()
            file_info.download(out=buffered)
            image_data = np.array(Image.open(buffered))
        except Exception:
            logging.warning(f"{self.__class__.__name__}: error while get numpy image with file_id: {file_id}. {traceback.format_exc()}")

        return image_data

    def send_numpy_image(self, image_data: np.ndarray, image_type: str, chat_id: int, context: CallbackContext):
        """
        Sends numpy image to chat with chat_id and context.

        :param image_data: numpy image
        :param image_type: send as "document" or as "photo"
        :param chat_id: id of chat
        :param context: context of message
        :return: True if success else False
        """

        assert image_type in ["document", "photo", "sticker"]
        try:
            if image_data is not None:
                buffered = io.BytesIO()

                img = Image.fromarray(image_data)
                # print(img.size)
                img.save(buffered, format="PNG")
                buffered.seek(0)
                if image_type == "document":
                    context.bot.send_document(chat_id=chat_id, document=buffered)
                if image_type == "photo":
                    context.bot.send_photo(chat_id=chat_id, photo=buffered)
                if image_type == "sticker":
                    context.bot.send_sticker(chat_id=chat_id, sticker=buffered)
                return True
        except Exception:
            logging.warning(f"{self.__class__.__name__}: error while send numpy image. {traceback.format_exc()}")
            return False

    def dump_image(self, image_data: np.ndarray, filename: str):

        path = os.path.join(self.dump_images_path, filename)
        skio.imsave(path, image_data)
        logging.info(f"{self.__class__.__name__}: dump to: {path}")

    # self.updater.bot.sendMessage(chat_id=key, text=message)
