import os
import pandas as pd

from ai_agent import ask_llama
import aiohttp
import discord
from discord import app_commands
from dotenv import load_dotenv

load_dotenv()
TOKEN = os.getenv("DISCORD_BOT_TOKEN")

# Optional: dev-only guild for instant slash command sync.
# Set DISCORD_GUILD_ID in your local .env. Leave it unset in production.
GUILD_ID_RAW = os.getenv("DISCORD_GUILD_ID")
DEV_GUILD = discord.Object(id=int(GUILD_ID_RAW)) if GUILD_ID_RAW else None

intents = discord.Intents.default()
intents.message_content = True
intents.dm_messages = True  # required to receive DMs

client = discord.Client(intents=intents)
tree = app_commands.CommandTree(client)

# Load shop history dataset
HISTORY = pd.read_csv("shop_history.csv", parse_dates=["appearance_date"])


@client.event
async def on_ready():
    if DEV_GUILD:
        # Dev mode: sync to one server instantly
        tree.copy_global_to(guild=DEV_GUILD)
        await tree.sync(guild=DEV_GUILD)
        print(f"Synced commands to dev guild {DEV_GUILD.id}")
    else:
        # Production: global sync (takes up to ~1 hour to propagate)
        await tree.sync()
        print("Synced commands globally (may take up to 1 hour to appear)")
    print(f"Logged in as {client.user}")


@client.event
async def on_message(message: discord.Message):
    if message.author.bot:
        return

    is_dm = message.guild is None
    is_mention = client.user in message.mentions

    # In servers: only reply when mentioned. In DMs: reply to everything.
    if not is_dm and not is_mention:
        return

    user_text = message.content
    if is_mention:
        user_text = user_text.replace(f"<@{client.user.id}>", "").strip()
        user_text = user_text.replace(f"<@!{client.user.id}>", "").strip()
    user_text = user_text.strip()

    if not user_text:
        await message.reply("Ask me about Fortnite shop rarity, item history, or today's shop 🙂")
        return

    async with message.channel.typing():
        try:
            response = ask_llama(user_text)
            await message.reply(response[:1900])
        except Exception as e:
            await message.reply(f"LLM error: `{e}`")


@tree.command(name="ping", description="Check if the bot is alive")
async def ping(interaction: discord.Interaction):
    await interaction.response.send_message("Pong! 🏓")


@tree.command(name="shop", description="Check the shop")
async def shop(interaction: discord.Interaction):
    await interaction.response.defer()
    async with aiohttp.ClientSession() as session:
        async with session.get("https://fortnite-api.com/v2/shop") as response:
            data = await response.json()
            cosmetics = []
            for entry in data["data"]["entries"]:
                if "brItems" in entry:
                    cosmetics.append(entry)
                    if len(cosmetics) == 10:
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


@tree.command(name="lookup", description="Show shop history for a cosmetic")
@app_commands.describe(name="Cosmetic name (e.g. Renegade Raider)")
async def lookup(interaction: discord.Interaction, name: str):
    await interaction.response.defer()

    matches = HISTORY[HISTORY["name"].str.lower().str.contains(name.lower(), na=False)]
    if matches.empty:
        await interaction.followup.send(f"No cosmetic found matching '{name}'.")
        return

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