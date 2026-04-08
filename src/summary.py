import pandas as pd
import sys
import os
from groq import Groq

# Get file path from argument
file_path = sys.argv[1]

# Load dataset
df = pd.read_csv(file_path)

# Basic calculations
total_reps = len(df)
good_reps = len(df[df["is_good"] == True])
bad_reps = total_reps - good_reps

# Most common mistake
mistakes = df[df["feedback"] != "Good form"]["feedback"]

if len(mistakes) > 0:
    common_mistake = mistakes.value_counts().idxmax()
else:
    common_mistake = "None"

# Print basic summary
print("\n=== SESSION SUMMARY ===")
print(f"Total reps: {total_reps}")
print(f"Good reps: {good_reps}")
print(f"Bad reps: {bad_reps}")
print(f"Most common mistake: {common_mistake}")


# Groq client (use env variable)
api_key = os.getenv("GROQ_API_KEY")

if not api_key:
    print("\n⚠️ AI feedback skipped: GROQ_API_KEY not set")
    sys.exit()

client = Groq(api_key=api_key)

prompt = f"""
You are a fitness coach.

User workout summary:
- Exercise: {df['exercise'][0]}
- Total reps: {total_reps}
- Good reps: {good_reps}
- Bad reps: {bad_reps}
- Common mistake: {common_mistake}

Give short, clear improvement advice (2-3 lines).
"""

response = client.chat.completions.create(
    model="llama3-8b-8192",
    messages=[{"role": "user", "content": prompt}]
)

print("\n=== AI COACH FEEDBACK ===")
print(response.choices[0].message.content)