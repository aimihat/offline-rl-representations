"""Run in background: uploads vids to GDrive"""

import glob
import os
import time

from pydrive.auth import GoogleAuth
from pydrive.drive import GoogleDrive

games = {"Pong": 0, "YarsRevenge": 0, "Enduro": 0, "Breakout": 0}

VIDEOS_PATH = os.environ.get("ATARI_REPLAYS_PATH")

gauth = GoogleAuth()

gauth.LoadCredentialsFile("credentials.txt")
if gauth.credentials is None:
    gauth.CommandLineAuth()
elif gauth.access_token_expired:
    gauth.Refresh()
else:
    gauth.Authorize()
gauth.SaveCredentialsFile("credentials.txt")

drive = GoogleDrive(gauth)

fileList = drive.ListFile(
    {"q": "'16WZuTfldaU-GI8DBnNHkgT2beEeWdW3o' in parents and trashed=false"}
).GetList()
for f in fileList:
    if f["title"] in games.keys():
        games[f["title"]] = f["id"]

print(f"Checking for files...")
while True:
    # Upload all new videos to their Drive folder and delete them
    for f in glob.glob(VIDEOS_PATH + "*"):
        # Get folder for that game
        print(f"found {f}")
        fileid = games[f.split("/")[-1].split("-")[0]]
        file1 = drive.CreateFile(
            {
                "mimeType": "video/mp4",
                "parents": [{"kind": "drive#fileLink", "id": fileid}],
            }
        )
        file1.SetContentFile(f)
        file1.Upload()

        os.remove(f)
        print(f"{f} was uploaded and deleted")

    time.sleep(10)
