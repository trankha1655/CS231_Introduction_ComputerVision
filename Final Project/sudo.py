class sudoku:
    def __init__(self, array_sudoku):
        self.bo = array_sudoku
    def find_empty(self):
        for i in range(len(self.bo)):
            for j in range(len(self.bo[0])):
                if self.bo[i][j] == 0:
                    return (i, j)
        return None

    def valid(self, num, pos):
        for i in range(len(self.bo[0])):
            if self.bo[pos[0]][i] == num and pos[1] != i:
                return False
        for i in range(len(self.bo)):
            if self.bo[i][pos[1]] == num and pos[0] != i:
                return False
        box_x = pos[1] // 3
        box_y = pos[0] // 3
        for i in range(box_y * 3, box_y * 3 + 3):
            for j in range(box_x * 3, box_x * 3 + 3):
                if self.bo[i][j] == num and (i, j) != pos:
                    return False
        return True

    def solve(self):
        find = self.find_empty()
        if not find:
            return True
        else:
            row, col = find
        for i in range(1, 10):
            if self.valid(i, (row, col)):
                self.bo[row][col] = i
                if self.solve():
                    return True
                self.bo[row][col] = 0
        return False
