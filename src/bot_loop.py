import logging
import src.config as cfg 
from datetime import datetime, timezone
from telegram.ext import Application
from src.bot.handlers import register_handlers

logging.basicConfig(
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    level=logging.INFO
)

logger = logging.getLogger(__name__)

def main():
    token = cfg.TG_TOKEN
    if not token:
        logger.error("Error: Telegram bot token not found.")
        return
    
    bot_startup_time = datetime.now(timezone.utc)

    try:
        app = Application.builder().token(token).concurrent_updates(True).build()
        register_handlers(app, bot_startup_time)
        logger.info("Bot started.")
        app.run_polling()
    except Exception as e:
        logger.exception("An error occurred while running the bot: %s", e)
        

if __name__ == "__main__":
    main()
