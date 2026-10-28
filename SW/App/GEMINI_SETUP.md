# Gemini API Setup

The Flutter Web application reads the Gemini API key from a Dart environment variable.

Do not commit an actual API key to this repository.

## Windows PowerShell

Set the API key as a user environment variable:

[Environment]::SetEnvironmentVariable("GEMINI_API_KEY", "YOUR_GEMINI_API_KEY", "User")

Open a new PowerShell window and verify:

if ($env:GEMINI_API_KEY) { "GEMINI_API_KEY OK" } else { "GEMINI_API_KEY NOT FOUND" }

Build Flutter Web:

cd C:\Users\82106\robot_arm_app
flutter build web --release --dart-define=GEMINI_API_KEY="$env:GEMINI_API_KEY"

The application reads the value with:

const apiKey = String.fromEnvironment('GEMINI_API_KEY');

Never commit the actual API key to GitHub.
