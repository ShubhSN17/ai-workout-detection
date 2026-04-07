import subprocess

def run_script(script_name):
    try:
        result = subprocess.run(
            ["python", f"Exercises/{script_name}"],
            capture_output=True,
            text=True
        )

        print(result.stdout)

        # Detect session log path
        for line in result.stdout.split("\n"):
            if "SESSION_LOG:" in line:
                log_file = line.split("SESSION_LOG:")[-1].strip()

                # Run summary
                subprocess.run(["python", "src/summary.py", log_file])

    except Exception as e:
        print("Error running script:", e)

def main():
    while True:
        print("\n=== AI WORKOUT COACH ===")
        print("1. Squat")
        print("2. Push-up")
        print("3. Bicep Curl")
        print("4. Shoulder Press")
        print("5. Exit")

        choice = input("Select exercise: ")

        if choice == "1":
            run_script("squat.py")

        elif choice == "2":
            run_script("pushups.py")

        elif choice == "3":
            run_script("bicep.py")

        elif choice == "4":
            run_script("shoulder.py")

        elif choice == "5":
            print("Exiting...")
            break

        else:
            print("Invalid choice. Try again.")

if __name__ == "__main__":
    main()