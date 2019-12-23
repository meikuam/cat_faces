import os
import time
import random
import logging

from telegram.ext import CallbackContext
from src.text_processor.text_processor import TextProcessor


class DumbTextProcessor(TextProcessor):
    """
    """
    def __init__(self, dump_messages):
        super(DumbTextProcessor, self).__init__()
        self.dump_messages = dump_messages

    def __call__(self, chat_id: (str, int),  message: str, context: CallbackContext):
        """
        make some process with text and return answer to text
        """

        logging.info(f"{self.__class__.__name__}: message from:{chat_id} text:{message}")

        responses = [
            'пик пик пик',
            'что-то не выходит говорить',
            'мяу',
            'я не понимаю :с',
        ]
        ghost_in_the_shell_references = [
            'прошлого не изменишь, так что пусть оно остается со мной до конца',
            'я — живое мыслящее существо, рожденное в океане информации',
            'удивительно, как много энергии вкладывают люди в создание своего подобия',
            'человеку не дано знать, понимает он что-то или нет. он может только надеяться, что понимает',
            'не лежи в кровати, как труп. даже великий Конфуций говорил, что нельзя притворяться, будто заснул вечным сном',
            'будь мы все полностью одинаковыми, наши действия были бы полностью предсказуемыми, а ведь зачастую из ситуации бывает несколько выходов',
            'всё в этом мире меняется. желая остаться таким же ты лишь ограничиваешь себя',
            'искусство подражает жизни, жизнь тоже может подражать искусству'
        ]
        blade_runner_reference = [
            'я видел такое, что вам, людям, и не снилось...'
        ]
        responses = responses + ghost_in_the_shell_references + blade_runner_reference

        response_id = random.randint(0, len(responses) - 1)
        response_text = responses[response_id]
        context.bot.send_message(chat_id=chat_id, text=response_text)
        logging.info(f"{self.__class__.__name__}: message to:{chat_id} text:{response_text}")
        # if blade runner reference
        if response_id == len(responses) - 1:
            responses = [
                'атакующие корабли, пылающие над Орионом',
                'лучи Си, пронизывающие мрак близ ворот Тангейзера...',
                'все эти мгновения затеряются во времени, как слёзы в дожде.',
                'время... умирать'
            ]
            for i in range(len(responses)):
                if random.random() > 0.2:
                    context.bot.send_message(chat_id=chat_id, text="ой")
                    logging.info(f"{self.__class__.__name__}: message to:{chat_id} text: ой")
                    break
                time.sleep(random.random())
                context.bot.send_message(chat_id=chat_id, text=responses[i])
                logging.info(f"{self.__class__.__name__}: message to:{chat_id} text:{responses[i]}")
