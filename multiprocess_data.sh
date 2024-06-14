for i in `seq 1 32`;
do
  echo worker $i
  # on cloud:
  python generate_data.py &
  # on macbook for debugging:
  #python extract.py &
  sleep 1.0
done