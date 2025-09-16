class count_time:
    def __init__(self, start):
        self.start_time = start
        self.end_time = 0

    def get_end_time(self,t):
        self.end_time = t

    def get_total_time(self):
        return self.end_time - self.start_time