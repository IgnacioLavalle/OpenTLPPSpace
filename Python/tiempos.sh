#!/bin/bash


echo "===== Grafo pesado dataset completo ====="

for i in {1..4}
do
    echo "===== DDNE Corrida $i ====="
    python3 DDNE_fc_oos.py 2>&1 | tee "tiempos/ddne_corrida${i}.txt"
done

for i in {1..4}
do
    echo "===== ELD Corrida $i ====="
    python3 eld_fc.py 2>&1 | tee "tiempos/eld_corrida${i}.txt"
done

for i in {1..4}
do
    echo "===== d2v Corrida $i ====="
    python3 d2v_fc.py 2>&1 | tee "tiempos/d2v_corrida${i}.txt"
done

for i in {1..4}
do
    echo "===== TMF Corrida $i ====="
    python3 TMF_fc.py 2>&1 | tee "tiempos/tmf_corrida${i}.txt"
done

for i in {1..4}
do
    echo "===== List Corrida $i ====="
    python3 LIST_FC.py 2>&1 | tee "tiempos/list_corrida${i}.txt"
done



echo "===== Grafo no pesado dataset completo ====="

for i in {1..4}
do
    echo "===== TMF no pesado Corrida $i ====="
    python3 tmf_uw_fc.py 2>&1 | tee "tiempos/tmy_unw_corrida${i}.txt"
done

for i in {1..4}
do
    echo "===== List no pesado Corrida $i ====="
    python3 list_uw_fc.py 2>&1 | tee "tiempos/list_unw_corrida${i}.txt"
done



echo "===== Grafo pesado dataset recortado ====="


for i in {1..4}
do
    echo "===== DDNE rec Corrida $i ====="
    python3 gcngan_fc_oos.py 2>&1 | tee "tiempos/ddne_rec_corrida${i}.txt"
done

for i in {1..4}
do
    echo "===== DDNE rec Corrida $i ====="
    python3 DDNE_fc_oos.py --data_name 'Recortado677' 2>&1 | tee "tiempos/ddne_rec_corrida${i}.txt"
done

for i in {1..4}
do
    echo "===== ELD rec Corrida $i ====="
    python3 eld_fc.py --data_name 'Recortado677' 2>&1 | tee "tiempos/eld_rec_corrida${i}.txt"
done

for i in {1..4}
do
    echo "===== d2v rec Corrida $i ====="
    python3 d2v_fc.py --data_name 'Recortado677' 2>&1 | tee "tiempos/d2v_rec_corrida${i}.txt"
done

for i in {1..4}
do
    echo "===== TMF rec Corrida $i ====="
    python3 TMF_fc.py --data_name 'Recortado677' 2>&1 | tee "tiempos/tmf_rec_corrida${i}.txt"
done

for i in {1..4}
do
    echo "===== List rec Corrida $i ====="
    python3 LIST_FC.py --data_name 'Recortado677' 2>&1 | tee "tiempos/list_rec_corrida${i}.txt"
done

