#!/bin/bash

echo "🔄 Staging all changes..."
git add .
git add -u

echo "✅ Changes staged. Here's the status:"
git status

echo ""
read -p "📝 Enter commit message (e.g., 'Update: improved training accuracy'): " commit_msg

git commit -m "$commit_msg"

echo ""
read -p "🚀 Ready to push to GitHub? Press [ENTER] to continue..."

git push origin main

echo "✅ All done! Latest changes pushed to GitHub 🚀"

