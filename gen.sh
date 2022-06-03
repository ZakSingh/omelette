until python pretrain.py --generate True; do
    echo "generator crashed with exit code $?.  Respawning.." >&2
    sleep 1
done