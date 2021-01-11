# Data : http://semanticweb.kaist.ac.kr/home/index.php/KAIST_Corpus (1,000,000 phrases High quality morpho-syntactically annotated corpus)

rm -f ../KAIST/all.txt
for file in ../KAIST_raw/*.vrt ; 
do echo $file;  python3 convert.py -f $file >> ../KAIST/all.txt ; 
done
