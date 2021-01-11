rm -f ./temp.txt
for file in ../KMOU_raw/*.txt ; 
do echo $file;  python3 correction.py -g $file >> ./temp.txt ; 
done
#cat ./temp.txt | sed "s/^\w*$/_ _ _ _/g" > ../KMOU/all.txt
cat ./temp.txt | sed "s/^$/_ _ _ _/g" > ../KMOU/all.txt
