from tqdm import tqdm

if __name__ == '__main__':
    user = set()
    with open('tij_InVS.dat', 'r') as f:
        for line in f:
            for u in line.strip().split(' ')[1:]:
                user.add(u)
    print(len(user))