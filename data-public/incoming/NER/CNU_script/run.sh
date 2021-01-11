# Data :  https://github.com/dsindex/ntagger/tree/master/data/clova2019

#cat ../CNU_raw/*.txt > ./all.txt
cat ../CNU_raw/*.txt | sed "s/^$/_ _ _ _/g" > ../CNU/all.txt
