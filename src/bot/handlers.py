from datetime import timezone

from telegram import (
    Update,
    InlineKeyboardButton,
    InlineKeyboardMarkup, 
)

from telegram.ext import (
    ContextTypes,
    CommandHandler,
    MessageHandler,
    filters,
    CallbackQueryHandler,
)

from src.db.db import save_rating
from src.rag import query_rag_system


async def time_check(update: Update, context: ContextTypes.DEFAULT_TYPE) -> bool:
    bot_startup_time = context.bot_data.get('bot_startup_time')
    if update.message:
        message_time = update.message.date.replace(tzinfo=timezone.utc)
    elif update.callback_query:
        message_time = update.callback_query.message.date.replace(tzinfo=timezone.utc)
    else:
        return False
    return message_time >= bot_startup_time


# Ignore old messages
def validate_message_time(handler_func):
    async def wrapper(update: Update, context: ContextTypes.DEFAULT_TYPE, *args, **kwargs):
        if not await time_check(update, context):
            return  
        return await handler_func(update, context, *args, **kwargs)
    return wrapper


@validate_message_time
async def start(update: Update, context: ContextTypes.DEFAULT_TYPE):
    user = update.effective_user
    await update.message.reply_html(
        rf"Hello {user.mention_html()}! Ask me anything."
    )


@validate_message_time
async def handle_message(update: Update, context: ContextTypes.DEFAULT_TYPE):   
    user_message = update.message.text
    response = query_rag_system(user_message)
    
    keyboard = [
        [
            InlineKeyboardButton("Good", callback_data='good'),
            InlineKeyboardButton("Bad", callback_data='bad'),
        ]
    ]
    reply_markup = InlineKeyboardMarkup(keyboard)
    
    message = await context.bot.send_message(
        chat_id=update.effective_chat.id,
        text=response,
        reply_markup=reply_markup
    )
    
    context.user_data['last_response'] = {
        'user_id': update.effective_user.id,
        'message_id': message.message_id,
        'query': user_message,
        'response_text': response
    }


@validate_message_time
async def handle_rating(update: Update, context: ContextTypes.DEFAULT_TYPE):
    query = update.callback_query
    rating = query.data
    await query.answer()
    
    response_info = context.user_data.get('last_response')
    if response_info:
        user_id = response_info['user_id']
        message_id = response_info['message_id']
        query = response_info['query']
        response_text = response_info['response_text']
        
        # Save the rating to db
        db_path = context.bot_data.get('ratings_db_path')
        save_rating(user_id, message_id, query, response_text, rating, db_path)
        
        await context.bot.send_message(
            chat_id=update.effective_chat.id,
            text="Thank you for your feedback!"
        )
    else:
        await context.bot.send_message(
            chat_id=update.effective_chat.id,
            text="Sorry, couldn't associate your rating with the response."
        )


def register_handlers(application, bot_startup_time, ratings_db_path):
    application.bot_data['bot_startup_time'] = bot_startup_time
    application.bot_data['ratings_db_path'] = ratings_db_path
    
    application.add_handler(CommandHandler('start', start))
    application.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, handle_message))
    application.add_handler(CallbackQueryHandler(handle_rating))
   