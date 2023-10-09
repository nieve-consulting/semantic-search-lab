WANDS_DOWNLOADS_PATH=/data/wands

# delete any old copies of temp data
rm -rf $WANDS_DOWNLOADS_PATH
 
# make directory for temp tiles
mkdir -p $WANDS_DOWNLOADS_PATH
 
# move to temp directory
cd $WANDS_DOWNLOADS_PATH
 
# download datasets
wget -q https://raw.githubusercontent.com/wayfair/WANDS/main/dataset/label.csv
wget -q https://raw.githubusercontent.com/wayfair/WANDS/main/dataset/product.csv
wget -q https://raw.githubusercontent.com/wayfair/WANDS/main/dataset/query.csv
 
# show folder contents
echo "Downloaded dataset files to $WANDS_DOWNLOADS_PATH"
ls -l