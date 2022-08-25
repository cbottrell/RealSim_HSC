cat > coords.txt <<EOT
# ra         dec             outfile(optional)
269.6323064579072 66.63089594376515 a.png
EOT

./.venv/bin/python colorPostage.py --user michitaro.nike@gmail.com --outDir png3 ./coords.txt
