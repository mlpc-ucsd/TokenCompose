
set -e

wget http://images.cocodataset.org/zips/train2017.zip
unzip train2017.zip
rm train2017.zip

python move_img.py

