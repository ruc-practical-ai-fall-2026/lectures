echo "Starting http-server..."

http-server

if [ $? -eq 0 ]; then
    echo "Closed http-server successfully."
else
    echo "Closed http-server unsuccessfully."
    exit 1
fi