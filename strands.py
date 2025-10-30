from collections import defaultdict
import itertools
from pathlib import Path

class PrefixTree(object):

    r"""
    Prefix Tree or Trie class. Creates a tree structure where each node is a letter.
    Traversing the tree to a leaf will spell a unique valid word.

    For example this is a PrefixTree containing 3 words: BAT, TREE, and TRY
         ROOT
        /    \ 
        B     T
        |     |
        A     R
        |    / \  
        T    E  Y
             |
             E
    
    Each node contains the tree structure below it. Each node also has an
    attribute "prefix" that contains the letters in the path followed to reach
    that node. A boolean value "is_word" denotes that this nodes prefix is a 
    valid word. Valid words may also have children. For example, RUN is a valid 
    word with child words RUNS, RUNNING, RUNNER, etc.
    """

    def __init__(self, words: list[str]):
        self.prefix = '' # Root prefix is empty string
        self.is_word = False
        self.children = defaultdict(lambda: self.__class__([]))
        for word in words:
            self.add_word(word, depth=0)

    def add_word(self, word: str, depth: int = 0):
        """
        Add a word segment to this node.

        Arguments:
            word: The complete word being added
            n: The current depth of this node
        """
        self.prefix = word[:depth]
        if self.prefix == word:
            self.is_word = True
        else:
            self.children[word[depth]].add_word(word, depth=depth+1)
    
    def __getitem__(self, prefix: str):
        """
        Traverse tree downwards following letters in prefix. Return a sub-tree
        containing words starting with the prefix, or KeyError if no such words
        exist in the tree.

        For example, for a prefix tree loaded with common English words.
        pt['HEL'] returns a sub-tree containing all words starting with HEL
        pt['ZZZWXQK'] raises a KeyError
        """
        if prefix[0] in self.children:
            if len(prefix) <= 1:
                return self.children[prefix[0]]
            else:
                return self.children[prefix[0]][prefix[1:]]
        else:
            raise KeyError(prefix)
    
    def __contains__(self, prefix: str):
        """
        Check if words beginning with prefix exist in the tree, including if
        prefix is a leaf node.
        """
        if len(prefix) <= 1:
            return prefix in self.children
        else:
            return prefix[1:] in self.children[prefix[0]]

    def get(self, *args):
        """
        Emulates dict.get
        """
        try:
            self[args[0]]
        except KeyError:
            return args[1] if len(args)>=2 else None

    def __iter__(self):
        """
        Yield all valid words from this node
        """
        if self.is_word:
            yield self
        for child in self.children.values():
            yield from child

    def __len__(self):
        """
        Count of valid words from this node
        """
        #TODO prepopulate during self.add
        n = int(self.is_word)
        return n + sum([len(_) for _ in self.children.values()])


class StrandsPuzzle():

    def __init__(self, grid: list[list[str]]):
        self._grid = grid
        self._dim = [len(self._grid), len(self._grid[0])]

    @classmethod
    def from_file(cls, fp: Path):
        grid = []
        for line in fp.open():
            grid.append([_ for _ in line.strip()])
        return cls(grid)

    def get_neighbor_coordinates(self, coord: tuple[int,int]) -> list[tuple[int,int]]:
        """
        Get coordinates of grid positions around the current one (adjacent or diagonal)
        """
        i, j = coord
        neighbors = []
        # Iterate the 9 positions near provided coordinates
        for ni, nj in itertools.product((i-1, i, i+1), (j-1,j,j+1)):
            # Drop the provided position
            if ni == i and nj == j:
                continue
            # Drop out-of-bounds
            if ni < 0 or ni >= self._dim[0]:
                continue
            if nj < 0 or nj >= self._dim[1]:
                continue
            neighbors.append((ni,nj))
        return neighbors    

    def __getitem__(self, coord: tuple[int,int]) -> str:
        """
        Get the letter at a coordinate
        """
        return self._grid[coord[0]][coord[1]]

    def find_words(self, coords: list[tuple[int,int]], prev_pt: PrefixTree) -> tuple[str,list[tuple[int,int]]]:
        """
        Main puzzle solving function.

        Arguments:
            coords: The coordinates of the path taken to get to this position,
                and the coordinate of the current position. This is to avoid
                using positions more than once in a word.
            prev_pt: The node in the prefix tree up to but not including the
                current letter. Thus, prev_pt.prefix is the word so far.
        """
        cur_coord = coords[-1]
        cur_let = self[cur_coord]
        words = []
        # Proceed if adding cur_let will continue to a valid word
        if cur_let in prev_pt:
            cur_pt = prev_pt[cur_let]
            # Check for complete word
            if cur_pt.is_word:
                words.append((cur_pt.prefix, coords))
            # Proceed to check neighboring positions
            for neighbor_coord in self.get_neighbor_coordinates(cur_coord):
                # Can only use a position once
                if neighbor_coord in coords:
                    continue
                # Recur into next position
                words += self.find_words(coords+[neighbor_coord], cur_pt)
        return words    


if __name__ == '__main__':
    from argparse import ArgumentParser

    parser = ArgumentParser()
    parser.add_argument('puzzle',
        type = Path,
        help = 'Path to puzzle file',
    )
    parser.add_argument('--words',
        type = Path,
        required = False,
        default = Path('words_easy.txt'),
        help = 'Path to words file',
    )
    parser.add_argument('--show',
        type = int,
        default = 10,
        help = 'Number of words to show',
    )
    args = parser.parse_args()

    all_words = [l.strip().upper() for l in args.words.open()]
    # Words must have more than 3 letters
    words = filter(lambda _: len(_) > 3, all_words)
    pt = PrefixTree(words)

    puzzle = StrandsPuzzle.from_file(args.puzzle)

    solutions = []
    found_words = set()
    # Iterate every location
    for i,j in itertools.product(range(0,puzzle._dim[0]), range(0,puzzle._dim[1])):
        coord = (i,j)
        for solution in puzzle.find_words([coord], pt):
            # Many words have multiple valid paths, especially at start of game
            if solution[0] not in found_words:
                solutions.append(solution)
                found_words.add(solution[0])
    # Order words longest to shortest
    solutions.sort(key=lambda _: (len(_[0]), _[1]), reverse=True)
    # Print some number of words and their starting coordinate
    for solution in solutions[:args.show]:
        print(solution[0], solution[1][0])
