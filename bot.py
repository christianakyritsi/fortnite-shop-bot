import os
import pandas as pd

import aiohttp
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

# Load shop history dataset
HISTORY=pd.read_csv("shop_history.csv", parse_dates=["appearance_date"])

@client.event
async def on_ready():
    await tree.sync(guild=GUILD_ID)
    print(f"Logged in as {client.user}")

@tree.command(name="ping", description="Check if the bot is alive", guild=GUILD_ID)
async def ping(interaction: discord.Interaction):
    await interaction.response.send_message("Pong! 🏓")

@tree.command(name="shop", description="Check the shop", guild=GUILD_ID)
async def shop(interaction: discord.Interaction):
    await interaction.response.defer()
    async with aiohttp.ClientSession() as session:
        async with session.get("https://fortnite-api.com/v2/shop") as response:
            data = await response.json()
            cosmetics = []
            for entry in data["data"]["entries"]:
                if "brItems" in entry:
                    cosmetics.append(entry)
                    if len(cosmetics)==10:
                        break
            embed = discord.Embed(title="Today's Shop")
            for entry in cosmetics:
                current_id = entry["brItems"][0]["id"]
                item_history = HISTORY[HISTORY["item_id"] == current_id]
                if item_history.empty:
                    rarity = "unknown"
                    scarcity = "✨ never in shop before"
                else:
                    rarity = item_history.iloc[0]["rarity"]
                    today = pd.Timestamp.now(tz="UTC")
                    past = item_history[item_history["appearance_date"] < today.normalize()]
                    if past.empty:
                        scarcity = "✨ debut — first time in shop"
                    else:
                        last_seen = past["appearance_date"].max()
                        days_since = (today - last_seen).days
                        day_word = "day" if days_since == 1 else "days"
                        scarcity = f"last seen {days_since} {day_word} ago"
                embed.add_field(
                    name=entry["brItems"][0]["name"],
                    value=f'{entry["finalPrice"]} V-Bucks • {rarity} • {scarcity}',
                    inline=False,
                )
            await interaction.followup.send(embed=embed)

@tree.command(name="lookup", description="Show shop history for a cosmetic", guild=GUILD_ID)
@app_commands.describe(name="Cosmetic name (e.g. Renegade Raider)")
async def lookup(interaction: discord.Interaction, name: str):
    await interaction.response.defer()

    matches = HISTORY[HISTORY["name"].str.lower().str.contains(name.lower(), na=False)]

    if matches.empty:
        await interaction.followup.send(f"No cosmetic found matching '{name}'.")
        return

    # If multiple items match, pick the one with the most appearances
    top_id = matches["item_id"].value_counts().idxmax()
    matches = matches[matches["item_id"] == top_id]

    first_seen = matches["appearance_date"].min()
    last_seen = matches["appearance_date"].max()
    total = len(matches)
    rarity = matches.iloc[0]["rarity"]
    item_type = matches.iloc[0]["type"]

    today = pd.Timestamp.now(tz="UTC")
    days_since = (today - last_seen).days
    day_word = "day" if days_since == 1 else "days"

    embed = discord.Embed(
        title=matches.iloc[0]["name"],
        description=f"{rarity} {item_type}",
        color=0x4a90e2,
    )
    embed.add_field(name="Total appearances", value=str(total), inline=True)
    embed.add_field(name="First seen", value=first_seen.strftime("%Y-%m-%d"), inline=True)
    embed.add_field(name="Last seen", value=f"{last_seen.strftime('%Y-%m-%d')} ({days_since} {day_word} ago)", inline=True)

    await interaction.followup.send(embed=embed)

if __name__ == "__main__":
    if not TOKEN:
        raise RuntimeError(
        "DISCORD_BOT_TOKEN not found. "
        "Make sure your .env file defines DISCORD_BOT_TOKEN"
        )
    client.run(TOKEN)
