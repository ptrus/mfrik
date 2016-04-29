from random import randint
import linecache

def shuffle_file(in_path, out_path, n_lines, header=True):
    with open(out_path, 'w') as out:
        numbers = list(range(n_lines))
        # Copy header
        if header:
            out.write(linecache.getline(in_path, 1))
            n_lines -= 1

        for i in range(n_lines):
            idx = randint(0, n_lines-i-1)
            if header: idx += 1
            n = numbers.pop(idx)
            assert n != 0
            # Write line n, to output
            out.write(linecache.getline(in_path, n+1))

if __name__ == "__main__":
    #shuffle_file("/home/peterus/Downloads/ccdm_large.tsv", "/home/peterus/Downloads/ccdm_large-shuffled.tsv", 2450001)
    shuffle_file("C:\\Users\\Peter\\Downloads\\ccdm_large.tsv\\ccdm_large.tsv", "C:\\Users\\Peter\\Downloads\\ccdm_large.tsv\\ccdm_large-shuffled.tsv", 2450001)
