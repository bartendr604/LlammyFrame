# Replace just the sleep function
sed -i '' '/def is_sleeping_hours/,/return current_hour/c\
    def is_sleeping_hours(self):\
        """Check if it'\''s MY sleeping hours"""\
        current_hour = datetime.now().hour\
        return current_hour >= [YOUR_BEDTIME] or current_hour <= [YOUR_WAKE_TIME]' ~/Documents/llammy_trainer.py
mode#!/usr/bin/env python3
import sys

def run_single_session():
    print("🎬 Test visual session starting...")
    print("✅ Test complete!")

if __name__ == "__main__":
    if "--single-session" in sys.argv:
        run_single_session()
    else:
        print("🚀 Full trainer mode (use --single-session for test)")

Mode
Test

tail -10 ~/Documents
date
python3 ~/Documents/llammy_trainer.py --single-session
# Monitor for the next session
tail -f ~/Documents/llammy_training_log.txt
python3 -c "
from datetime import datetime
hour = datetime.now().hour
print(f'Current hour: {hour}')
print(f'Should be YouTube mode (11PM-7AM): {hour >= 23 or hour <= 7}')
print(f'Time: {datetime.now().strftime(\"%H:%M\")}')
"

ping -c 1 8.8.8.8

grep -E "(MODE|YouTube|OFFLINE)" ~/Documents/llammy_training_log.txt | tail -5
curl -o ~/Documents/llammy_visual_trainer.py https://claude.ai/artifacts/[artifact_id]
# Or copy manually from the artifact above
chmod +x ~/Documents/llammy_visual_trainer.py
python3 ~/Documents/llammy_visual_trainer.py --single-session
python3 ~/Documents/llammy_visual_trainer.py
# Watch for the next one
tail -f ~/Documents/llammy_training_log.txt
python3 ~/Documents/llammy_visual_trainer.py --single-session
# Stop the background one
launchctl unload ~/Library/LaunchAgents/com.llammy.trainer.plist

# Start the visual beast
python3 ~/Documents/llammy_visual_trainer.py
# Quick test with your existing trainer but visual mode
"/Applications/Blender 4.5.app/Contents/MacOS/Blender" --python /tmp/llammy_training_script.py

# Check if backup exists

python3 ~/Documents/llammy_visual_trainer.py




ls -la ~/Documents/llammy_trainer_backup.py

# If it exists, restore it
# See what's in the current trainer file
head -10 ~/Documents/llammy_trainer.py




ner.py

# Recreate the original background trainer
cat > ~/Documents/llammy_trainer.py << 'EOF'
# [Need to recreate the original background trainer]
EOF

# Edit the plist to remove --single-session and use visual trainer
nano ~/Library/LaunchAgents/com.llammy.trainer.plist/llammy_training_log.txt
# Save your visual trainer to its own file
cp ~/Documents/llammy_trainer.py ~/Documents/llammy_visual_trainer.py

# Create a modified background version for LaunchAgent
cat > ~/Documents/llammy_trainer.py << 'EOF'
# [Create a version that works with --single-session but runs in background]
EOF


# Test the visual trainer you have
python3 ~/Documents/llammy_trainer.py --single-session
# Check if visual trainer has --single-session support (it does!)
python3 ~/Documents/llammy_trainer.py --single-session
python3 ~/Documents/llammy_trainer.py --single-session
which python3
python3 --version
cd ~/Documents
python3 llammy_trainer.py --single-session
pwd
ls -la
python3 ~/Documents/llammy_trainer.py --single-session
# Press Ctrl+C to cancelcç
cat > ~/Documents/llammy_trainer.py << 'EOF'
# [Need to recreate the original background trainer]
EOF

launchctl list | grep llammy
# Edit the plist to work with visual trainer
cat > ~/Library/LaunchAgents/com.llammy.trainer.plist << 'EOF'
<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN" "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
<plist version="1.0">
<dict>
    <key>Label</key>
    <string>com.llammy.trainer</string>
    <key>ProgramArguments</key>
    <array>
        <string>python3</string>
        <string>/Users/jimmcquade/Documents/llammy_trainer.py</string>
        <string>--single-session</string>
    </array>
    <key>StartInterval</key>
    <integer>600</integer>
    <key>RunAtLoad</key>
    <true/>
</dict>
</plist>
EOF

launchctl unload ~/Library/LaunchAgents/com.llammy.trainer.plist
launchctl load ~/Library/LaunchAgents/com.llammy.trainer.plist
EOF
python3 ~/Documents/llammy_trainer.py
# Check if there's a syntax error
python3 -m py_compile ~/Documents/llammy_trainer.py

# Backup the current file
mv ~/Documents/llammy_trainer.py ~/Documents/llammy_trainer_broken.py

# Create a simple test version first
cat > ~/Documents/llammy_trainer.py << 'EOF'
#!/usr/bin/env python3
import sys

def run_single_session():
    print("🎬 Test visual session starting...")
    print("✅ Test complete!")

if __name__ == "__main__":
    if "--single-session" in sys.argv:
        run_single_session()
    else:
        print("🚀 Full trainer mode (use --single-session for test)")
EOF

chmod +x ~/Documents/llammy_trainer.py
cat > ~/Documents/llammy_trainer.py << 'EOF'
#!/usr/bin/env python3
python3 ~/Documents/llammy_trainer.py --single-session
EOF
chmod +x ~/Documents/llammy_trainer.py
python3 ~/Documents/llammy_trainer.py --single-session
cat > file << 'EOF'
cat > ~/Documents/llammy_trainer.py << 'EOF'
#!/usr/bin/env python3
import sys

def run_single_session():
    print("🎬 Test visual session starting...")
    print("✅ Test complete!")

if __name__ == "__main__":
    if "--single-session" in sys.argv:
        run_single_session()
    else:
        print("🚀 Full trainer mode (use --single-session for test)")


