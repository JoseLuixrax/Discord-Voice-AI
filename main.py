import discord
from discord.ext import commands, voice_recv
import os
import asyncio
from dotenv import load_dotenv
import openai
import time
import struct
import math

# Load environment variables
load_dotenv()
TOKEN = os.getenv('DISCORD_TOKEN')
OPENAI_API_KEY = os.getenv('OPENAI_API_KEY')

if not TOKEN:
    raise ValueError("No DISCORD_TOKEN found in .env file")
if not OPENAI_API_KEY:
    raise ValueError("No OPENAI_API_KEY found in .env file")

client = openai.OpenAI(api_key=OPENAI_API_KEY)

# Bot setup
intents = discord.Intents.default()
intents.message_content = True
bot = commands.Bot(command_prefix='!', intents=intents)

# Audio settings
SILENCE_THRESHOLD = 500  # RMS amplitude threshold for silence
SILENCE_DURATION = 1.2   # Seconds of silence to trigger response
MIN_AUDIO_LENGTH = 0.5   # Minimum duration of speech to process (seconds)

class ContinuousSink(voice_recv.AudioSink):
    def __init__(self, bot, channel):
        super().__init__()
        self.bot = bot
        self.channel = channel
        self.user_data = {} # {user_id: {'buffer': bytearray, 'last_spoken': time, 'processing': bool}}
        self.speaking_users = set()

    def wants_opus(self):
        return False  # We want PCM

    def write(self, user, data):
        if user is None or self.voice_client.is_playing():
            return

        pcm = data.pcm
        # Calculate RMS amplitude manually to avoid audioop dependency issues
        count = len(pcm) // 2
        format = "<" + "h" * count
        try:
            samples = struct.unpack(format, pcm)
            sum_squares = sum(s * s for s in samples)
            rms = math.sqrt(sum_squares / count) if count > 0 else 0
        except Exception:
            rms = 0

        now = time.time()
        
        if user.id not in self.user_data:
            self.user_data[user.id] = {
                'buffer': bytearray(),
                'last_spoken': now,
                'processing': False
            }
            
        user_record = self.user_data[user.id]

        if user_record['processing']:
            return # Ignore while processing

        if rms > SILENCE_THRESHOLD:
            user_record['buffer'].extend(pcm)
            user_record['last_spoken'] = now
            self.speaking_users.add(user.id)
        else:
            # Silence detected
            if user.id in self.speaking_users:
                # Was speaking, now silent. Check duration.
                silence_duration = now - user_record['last_spoken']
                if silence_duration > SILENCE_DURATION:
                    self.speaking_users.remove(user.id)
                    
                    # Check if buffer is long enough
                    if len(user_record['buffer']) > MIN_AUDIO_LENGTH * 48000 * 2 * 2:
                        # Process in background
                        audio_copy = bytes(user_record['buffer'])
                        user_record['processing'] = True
                        asyncio.run_coroutine_threadsafe(
                            self.process_audio(user, audio_copy), 
                            self.bot.loop
                        )
                    
                    # Clear buffer
                    user_record['buffer'] = bytearray()

    async def process_audio(self, user, pcm_data):
        try:
            print(f"Processing audio from {user.name}...")
            
            import wave
            filename = f"recording_{user.id}_{int(time.time())}.wav"
            with wave.open(filename, 'wb') as wav_file:
                wav_file.setnchannels(2)
                wav_file.setsampwidth(2)
                wav_file.setframerate(48000)
                wav_file.writeframes(pcm_data)

            # 1. STT (Whisper)
            try:
                with open(filename, "rb") as audio_file:
                    transcript = client.audio.transcriptions.create(
                        model="whisper-1", 
                        file=audio_file
                    )
                user_text = transcript.text
                print(f"{user.name} said: {user_text}")
            except Exception as e:
                print(f"Error in STT: {e}")
                return
            finally:
                if os.path.exists(filename):
                    os.remove(filename)

            if not user_text or len(user_text.strip()) == 0:
                return

            # 2. Chat (GPT)
            try:
                response = client.chat.completions.create(
                    model="gpt-3.5-turbo",
                    messages=[
                        {"role": "system", "content": "You are a helpful voice assistant in Discord. Keep answers concise and conversational."},
                        {"role": "user", "content": user_text}
                    ]
                )
                bot_reply = response.choices[0].message.content
                print(f"Reply: {bot_reply}")
            except Exception as e:
                print(f"Error in Chat: {e}")
                return

            # 3. TTS (OpenAI)
            try:
                tts_response = client.audio.speech.create(
                    model="tts-1",
                    voice="alloy",
                    input=bot_reply
                )
                tts_filename = f"reply_{int(time.time())}.mp3"
                tts_response.stream_to_file(tts_filename)
                
                # Play audio
                if self.voice_client.is_connected():
                    self.voice_client.play(discord.FFmpegPCMAudio(tts_filename), after=lambda e: self.cleanup_file(tts_filename))
                    
            except Exception as e:
                print(f"Error in TTS: {e}")

        except Exception as e:
            print(f"Error in process_audio: {e}")
        finally:
            if user.id in self.user_data:
                self.user_data[user.id]['processing'] = False

    def cleanup_file(self, filename):
        if os.path.exists(filename):
            os.remove(filename)

    def cleanup(self):
        pass

@bot.event
async def on_ready():
    print(f'Logged in as {bot.user.name}')

@bot.command()
async def join(ctx):
    """Joins the voice channel and starts listening."""
    if ctx.author.voice:
        channel = ctx.author.voice.channel
        if ctx.voice_client is not None:
            await ctx.voice_client.move_to(channel)
        else:
            await channel.connect(cls=voice_recv.VoiceRecvClient)
        
        vc = ctx.voice_client
        await ctx.send(f"Joined {channel} and started listening!")
        
        # Start listening
        vc.listen(ContinuousSink(bot, ctx.channel))
    else:
        await ctx.send("You are not in a voice channel.")

@bot.command()
async def leave(ctx):
    """Leaves the voice channel."""
    if ctx.voice_client:
        ctx.voice_client.stop_listening()
        await ctx.voice_client.disconnect()
        await ctx.send("Disconnected")
    else:
        await ctx.send("I am not in a voice channel.")

if __name__ == "__main__":
    bot.run(TOKEN)
