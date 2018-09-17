class Node:
    def __init__(self, id, childst, coverst):
        self.id = id
        self.childst = childst
        self.coverst = coverst

    def __str__(self):
        line = "id: %s\n" % self.id
        line += "childst: %s\n" % self.childst
        line += "coverst: %s\n" % self.coverst
        return line

