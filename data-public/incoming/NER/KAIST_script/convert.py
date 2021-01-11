from optparse import OptionParser
import sys 


if __name__ == '__main__':
    parser = OptionParser()
    parser.add_option("-f", "--filename", dest="filename", help="filename")
    (options, args) = parser.parse_args()
    fd = sys.stdout
    fe = sys.stderr

    content = False
    with open(options.filename,"r",encoding="utf8") as f:
        lines = f.readlines()
        for line in lines:
            if line.strip().endswith("</tdmsfiletext>") :
                content = False
            if content :  
                #l = l.replace('     ', '\t')
                l = line.rstrip(' \t\n\r')
                l = l.split('\t') 
                if line.strip():
                    if len(l[-1].split(' '))==1 : # missing index
                        l.extend(['-1','-1','-1','-1'])
                    else : 
                        l = l[:-1] + l[-1].split(' ')

                    '''
                    if 4<len(l)<7 : # whitespace to tab
                        idx = l[-4:]
                        ll = ['']
                        for c in ''.join(l[:-4]):
                            ll[-1] = ll[-1] + c
                            if (c == '\t') or (len(ll[-1])>=8):
                                fe.write('\n'+ll[-1]+'\n')
                                ll.append('')
                        ll = [x.strip() for x in ll if x.strip()] + idx
                        if (all([(' ' not in x) for x in ll])): # sanity check
                            l = ll
                    '''

                    if len(l) == 7 and all([x.isdigit() for x in l[-4:]]):
                        fd.write('\t'.join(l)+'\n')
                    elif ''.join(l).strip():
                        fe.write(f"ERROR : {line.strip()} : {'|'.join(l)}\n")
            if line.strip().endswith("<tdmsfiletext>") :
                content = True

