from collections import defaultdict
import itertools

class PrefixTree(object):
    def __init__(self, words):
        self.prefix = ''
        self.is_word = False
        self.children = defaultdict(lambda: self.__class__([]))
        for word in words:
            self.add(word, n=0)

    def add(self, word, n=0):
        self.prefix = word[:n]
        if self.prefix == word:
            self.is_word = True
        else:
            self.children[word[n]].add(word, n=n+1)
    
    def __getitem__(self, prefix):
        if prefix[0] in self.children:
            if len(prefix) <= 1:
                return self.children[prefix[0]]
            else:
                return self.children[prefix[0]][prefix[1:]]
        else:
            raise KeyError(prefix)
    
    def __contains__(self, prefix):
        if len(prefix) <= 1:
            return prefix in self.children
        else:
            return prefix[1:] in self.children[prefix[0]]

    def get(self, *args):
        try:
            self[args[0]]
        except KeyError:
            return args[1] if len(args)>=2 else None

    def __iter__(self):
        if self.is_word:
            yield self
        for child in self.children.values():
            yield from child

    def __len__(self):
        n = int(self.is_word)
        return n + sum([len(_) for _ in self.children.values()])


class StrandsPuzzle():

    def __init__(self, grid):
        self._grid = grid
        self._dim = [len(self._grid), len(self._grid[0])]

    @classmethod
    def from_file(cls, f):
        grid = []
        for line in f:
            grid.append([_ for _ in line.strip()])
        return cls(grid)

    def neighbors(self, coord):
        i, j = coord
        for ni, nj in itertools.product((i-1, i, i+1), (j-1,j,j+1)):
            if ni == i and nj == j:
                continue
            if ni < 0 or ni >= self._dim[0]:
                continue
            if nj < 0 or nj >= self._dim[1]:
                continue
            yield (ni, nj)        

    def __getitem__(self, coord):
        return self._grid[coord[0]][coord[1]]

    def find_words(self, coords, prev_pt):
        cur_coord = coords[-1]
        cur_let = self[cur_coord]
        if cur_let in prev_pt:
            children = []
            cur_pt = prev_pt[cur_let]
            if cur_pt.is_word and len(coords) > 3:
                children.append((cur_pt.prefix, coords))
            for adj_coord in self.neighbors(cur_coord):
                if adj_coord in coords:
                    continue
                next_coords = coords + [adj_coord]
                children += self.find_words(next_coords, cur_pt)
            return children
        else:
            return []        


if __name__ == '__main__':
    from argparse import ArgumentParser
    from pathlib import Path

    parser = ArgumentParser()
    parser.add_argument('puzzle',
        type = Path,
    )
    parser.add_argument('--words',
        type = Path,
        required = False,
        default = Path('words_easy.txt'),
    )
    args = parser.parse_args()

    with open(args.words) as f:
        words = [l.strip().upper() for l in f]
    pt = PrefixTree(words)

    with open(args.puzzle) as f:
        puzzle = StrandsPuzzle.from_file(f)

    solutions = []
    for i,j in itertools.product(range(0,puzzle._dim[0]), range(0,puzzle._dim[1])):
        coord = (i,j)
        solutions += puzzle.find_words([coord], pt)
    solutions.sort(key=lambda _: len(_[0]), reverse=True)
    for solution in solutions[:10]:
        print(solution[0], solution[1][0])
