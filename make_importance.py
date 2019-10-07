fwrite = open("importance.txt", "w")
with open("lower_0.6.txt") as fobj:
    for l in fobj:
        fwrite.write(l.strip()[:-7])
        fwrite.write("\n")
fwrite.close()