#!/usr/bin/env python
# vim: set fileencoding=utf-8 :

import logging
import urllib3
import urllib
import base64
import json
import md5
import os
import re
import sys
from lxml.etree import HTML
import time
from telegram import ChatAction, ParseMode, InlineKeyboardButton, InlineKeyboardMarkup
from telegram.ext import Updater, CommandHandler, MessageHandler, CallbackQueryHandler, Filters
from telegram.ext.dispatcher import run_async

t0 = time.time()
from pred import predict
print("Model prepared in %.2f s" % (time.time() - t0))

http = urllib3.PoolManager()

@run_async
def image(bot, update):

    #bot.sendChatAction(chat_id=update.message.chat_id, action=ChatAction.TYPING)

    m = update.message
    if m.photo:
        file_obj = m.photo[-1]
    elif m.document:
        file_obj = m.document
    else:
        m.reply_text("Please send me image as photo or file.")
        return 

    photo_file = bot.getFile(file_obj.file_id)
    data = http.request("GET", photo_file.file_path).data

    id = md5.md5(data).hexdigest()

    file("debug/%s.jpg" % id, "wb").write(data)

    resp, result = predict("debug/%s.jpg" % id)
    
    reply_markup = InlineKeyboardMarkup([[
        InlineKeyboardButton("⭕️", callback_data="%s/%d/Y" % (id, resp)),
        InlineKeyboardButton("❓", callback_data="%s/%d/?" % (id, resp)),
        InlineKeyboardButton("❌", callback_data="%s/%d/N" % (id, resp)),
    ]])

    m.reply_text(
        "This image is a %d-jigen one. (S = %.4f)\n" % (resp, result) + 
        "Please vote whether it is right.",
        quote=True,
        reply_markup=reply_markup,
    )

def feedback(bot, update):
    m = update.callback_query

    if not re.match(r"^[0-9A-Fa-f]{32}/[23]/[YN?]$", m.data):
        m.answer() # Malformed data
        return

    id, jigen, answer = m.data.split('/')
    try:
        dirname = {"Y": "right", "N": "wrong", "?": "unknown"}[answer]
    except:
        m.answer() # Malformed data
        return

    if not os.path.isfile("debug/%s.jpg" % id):
        m.answer("You already voted this image.")
        return

    os.rename("debug/%s.jpg" % id, "debug/%s/%s/%s.jpg" % (dirname, jigen, id))

    m.answer("Voted %s." % dirname)
    m.edit_message_reply_markup(reply_markup=None)

def error(bot, update, error):
    print(error)

def main():
    updater = Updater(sys.argv[1])

    updater.dispatcher.add_handler(MessageHandler(
        Filters.photo | 
        Filters.document |

        # These will be returned with error
        Filters.text |
        Filters.sticker
    , image))

    updater.dispatcher.add_handler(CallbackQueryHandler(feedback))
    updater.start_polling()

    updater.idle()

if __name__ == '__main__':
    main()
