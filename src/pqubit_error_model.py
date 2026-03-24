class pqubit_error_model:
    """
    Simple discrete model for mapping error rates to different actions of pqubits in surf patches.
    """

    after_clifford_depolarization: float
    before_round_data_depolarization: float
    before_measure_flip_probability: float
    after_reset_flip_probability: float

    def __init__(self,
                 after_clifford_depolarization: float = 0.0,
                 before_round_data_depolarization: float = 0.0,
                 before_measure_flip_probability: float = 0.0,
                 after_reset_flip_probability: float = 0.0
                 ):
        """
        :param after_clifford_depolarization:
        :param before_round_data_depolarization:
        :param before_measure_flip_probability:
        :param after_reset_flip_probability:
        """
        self.after_clifford_depolarization: float = after_clifford_depolarization
        self.before_round_data_depolarization: float = before_round_data_depolarization
        self.before_measure_flip_probability: float = before_measure_flip_probability
        self.after_reset_flip_probability: float = after_reset_flip_probability

    def get_after_clifford_depolarization(self) -> float:
        return self.after_clifford_depolarization

    def get_before_round_data_depolarization(self) -> float:
        return self.before_round_data_depolarization

    def get_before_measure_flip_probability(self) -> float:
        return self.before_measure_flip_probability

    def get_after_reset_flip_probability(self) -> float:
        return self.after_reset_flip_probability


    def set_after_clifford_depolarization(self, value: float) -> None:
        self.after_clifford_depolarization = value

    def set_before_round_data_depolarization(self, value: float) -> None:
        self.before_round_data_depolarization = value

    def set_before_measure_flip_probability(self, value: float) -> None:
        self.before_measure_flip_probability = value

    def set_after_reset_flip_probability(self, value: float) -> None:
        self.after_reset_flip_probability = value

    def set_error_rate(self, value: float) -> None:
        self.after_clifford_depolarization = value
        self.before_round_data_depolarization = value
        self.before_measure_flip_probability = value
        self.after_reset_flip_probability = value