file = open('Panda_Convo.txt','r')
f_sam = open('Samarth_words','a')
f_dum = open('Panda_words','a')
convo = file.readlines()
for line in convo:
    if "Samarth" in line:
        text = line.split('-')
        sam_sentence = text[1][text[1].index(':')+1:]
        f_sam.write(sam_sentence)

    elif "rachs" in line:
        text = line.split('-')
        sam_sentence = text[1][text[1].index(':') + 1:]
        f_dum.write(sam_sentence)

f_sam.close()
file.close()
f_dum.close()