import os

import discord
from discord import app_commands
from dotenv import load_dotenv

# Loading the .env file with the discord bot token
load_dotenv()

# Using capitals for constant variables 
TOKEN=os.getenv("DISCORD_BOT_TOKEN")

# Server id 
GUILD_ID=discord.Object(id=1499702607483109386)

# What kind of events the bot wants to receive
intents = discord.Intents.default()

# Object representing our bot's connection to Discord
# client handles the connection
client = discord.Client(intents=intents)

# Slash commands need to be stored in their own registry before being synced to discord
# tree is attached to our specific client
tree = app_commands.CommandTree(client)

@client.event
async def on_ready():
    tree.clear_commands(guild=None)
    await tree.sync()
    await tree.sync(guild=GUILD_ID)
    print(f"Logged in as {client.user}")

@tree.command(name ="ping", description="Check if the bot is alive", guild=GUILD_ID)
async def ping(interaction: discord.Interaction):
    await interaction.response.send_message("Pong! 🏓")


if __name__ == "__main__":
    if not TOKEN:
        raise RuntimeError(
        "DISCORD_BOT_TOKEN not found. "
        "Make sure your .env file defines DISCORD_BOT_TOKEN"
        )
    client.run(TOKEN)
