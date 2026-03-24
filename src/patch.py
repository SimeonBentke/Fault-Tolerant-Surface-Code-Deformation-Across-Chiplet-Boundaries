import warnings

from .pqubit import pqubit
from .qubit_topology import qubit_topology
from math import ceil
from typing import Callable
from stim import Circuit


class surf_patch(qubit_topology):
    """
    Surface-code patch to simulate in stim.
    """

    dx: int
    dz: int
    rounds: int
    basis: str
    init_state: str
    get_error_rate: Callable[[float], float]  # set inside self.set_uniform_error_rate\
    obs_qubits: list[pqubit]
    z_obs: list[pqubit]
    x_obs: list[pqubit]
    x_stab: list[pqubit]
    z_stab: list[pqubit]
    edges: dict[str,set[pqubit]]
    coupling_errors: dict[tuple[int,int],float]

    def __init__(
            self,
            dx: int,
            dz: int,
            r: int,
            basis: str,
            init_state: str,
            origin: tuple[float, float]=(0.0, 0.0),
            edge_types=None,
            uniform_error_rate=1e-3, id='',
            pqubit_id_offset=0
    ):
        """
        :param dx,dy: Code distance in x or y directions of the tile
        :param r: Number of QEC rounds
        :param basis: Basis to initialize the patch in
        :param init_state: State within basis to initialize patch in
        :param origin: Location of bottom left corner of the patch
        :param edge_types: Dict of stabilizer edge types of the patch
        :param uniform_error_rate: Error rate to homogenize physical operations with across the patch. If set to ``None``,
                will default to the independent physical error rates of each qubit
        :param id: Global ID of the patch (for use in the case of multiple patches)
        :param pqubit_id_offset: Value to offset all physical qubit IDs by
        """

        super().__init__(id, pqubit_id_offset, origin)

        self.dx: int = dx
        self.dz: int = dz
        self.rounds: int = r
        self.basis: str = basis
        self.init_state: str = init_state
        self.set_uniform_error_rate(uniform_error_rate)
        self.obs_qubits: list[pqubit] = []
        self.z_obs: list[pqubit] = []
        self.x_obs: list[pqubit] = []
        self.x_stab: set[pqubit] = set()
        self.z_stab: set[pqubit] = set()
        self.coupling_errors: dict[tuple[int,int],float] = {}

        if edge_types is None:
            self.edge_types: dict[str, str] = {'N': 'z', 'S': 'z', 'E': 'x', 'W': 'x'}
        else:
            self.edge_types: dict[str, str] = edge_types

        qubit_id_counter = pqubit_id_offset

        # Init data qubits
        for y in range(0, 2 * dz, 2):
            for x in range(0, 2 * dx, 2):
                pq = pqubit(qubit_id_counter, y + 1 + origin[0], x + 1 + origin[1], 'd', init_state)
                self.qubits.append(pq)
                self.data_qubits.add(pq)
                qubit_id_counter += 1

        # Init weight-4 X parity qubits
        for y in range(2, dx * 2, 2):

            init_x = 4 if y % 4 == 2 else 2

            for x in range(init_x, dx * 2 , 4):
                pq = pqubit(qubit_id_counter, y + origin[0], x + origin[1], 'x', '0')
                self.qubits.append(pq)
                self.parity_qubits.append(pq)

                qubit_id_counter += 1

                self.x_stab.add(pq)

        # Init weight-4 Z parity qubits
        for y in range(2, (dz * 2), 2):

            init_x = 4 if y % 4 == 0 else 2

            for x in range(init_x, (dx * 2), 4):
                pq = pqubit(qubit_id_counter, y + origin[0], x + origin[1], 'z', '0')
                self.qubits.append(pq)
                self.parity_qubits.append(pq)

                qubit_id_counter += 1

                self.z_stab.add(pq)

        # Init internal qubit/coordinate maps
        self._reset_qubit_maps()

        # North edge type
        y = (dz * 2)
        init_x = 2

        for x in range(init_x, (dx * 2), 2):

            if self.xy_map[(y-2  + origin[0], x + origin[1])].get_type() != self.edge_types['N']:
                pq = pqubit(qubit_id_counter, y + origin[0], x + origin[1], self.edge_types['N'], '0')
                self.qubits.append(pq)
                self.parity_qubits.append(pq)

                qubit_id_counter += 1

                if self.edge_types['N'] == 'z':
                    self.z_stab.add(pq)
                elif self.edge_types['N'] == 'x':
                    self.x_stab.add(pq)
                else:
                    raise ValueError('Unknown edge type')

        # South edge type
        y = 0
        init_x = 2

        for x in range(init_x, (dx * 2), 2):

            if self.xy_map[(y+2 + origin[0], x + origin[1])].get_type() != self.edge_types['S']:
                pq = pqubit(qubit_id_counter, y + origin[0], x + origin[1], self.edge_types['S'], '0')
                self.qubits.append(pq)
                self.parity_qubits.append(pq)

                qubit_id_counter += 1

                if self.edge_types['S'] == 'z':
                    self.z_stab.add(pq)
                elif self.edge_types['S'] == 'x':
                    self.x_stab.add(pq)
                else:
                    raise ValueError('Unknown edge type')

        # East edge type
        init_y = 2
        x = (dx*2)

        for y in range(init_y, (dz * 2), 2):

            if self.xy_map[(y + origin[0], x-2 + origin[1])].get_type() != self.edge_types['E']:
                pq = pqubit(qubit_id_counter, y + origin[0], x + origin[1], self.edge_types['E'], '0')
                self.qubits.append(pq)
                self.parity_qubits.append(pq)

                qubit_id_counter += 1

                if self.edge_types['E'] == 'z':
                    self.z_stab.add(pq)
                elif self.edge_types['E'] == 'x':
                    self.x_stab.add(pq)
                else:
                    raise ValueError('Unknown edge type')

        # West edge type
        init_y = 2
        x = 0

        for y in range(init_y, (dz * 2), 2):

            if self.xy_map[(y + origin[0], x+2 + origin[1])].get_type() != self.edge_types['W']:
                pq = pqubit(qubit_id_counter, y + origin[0], x + origin[1], self.edge_types['W'], '0')
                self.qubits.append(pq)
                self.parity_qubits.append(pq)

                qubit_id_counter += 1

                if self.edge_types['W'] == 'z':
                    self.z_stab.add(pq)
                elif self.edge_types['W'] == 'x':
                    self.x_stab.add(pq)
                else:
                    raise ValueError('Unknown edge type')

        # Init internal qubit/coordinate maps
        self._reset_qubit_maps()

        # The data qubits are the first qubits initialized, so qubits [0,dx*dy - 1] are the data qubits

        # Go between the two z type edges

        # North to South
        if self.edge_types['N'] == 'z' and self.edge_types['S'] == 'z':
            data_qubit = pqubit_id_offset + int(ceil(dx / 2)) - 1
            for y in range(dz):
                self.z_obs.append(self.id_map[data_qubit + (y * dx)])

        # East to West
        elif self.edge_types['E'] == 'z' and self.edge_types['W'] == 'z':
            data_qubit = pqubit_id_offset + int(dz / 2) * dx
            for x in range(dx):
                self.z_obs.append(self.id_map[data_qubit + (x)])

        # Bad edges
        else:
            """raise ValueError(
                            'Bad edge types; (North, South) must have the same edge types, and (East, West) must have the same edge types.')"""
            warnings.warn(
                "Cannot initialize observable qubits list. Either the North/South or East/West edge types do not match\n"
                "Explicitly declare the observables with <object>.x_obs:list[int] = [...] or <object>.z_obs:list[int] = [...]")

        # Go between the two x type edges

        # North to South
        if self.edge_types['N'] == 'x' and self.edge_types['S'] == 'x':
            data_qubit = pqubit_id_offset + int(ceil(dx / 2)) - 1
            for y in range(dz):
                self.x_obs.append(self.id_map[data_qubit + (y * dx)])

        # East to West
        elif self.edge_types['E'] == 'x' and self.edge_types['W'] == 'x':
            data_qubit = pqubit_id_offset + int(dz / 2) * dx
            for x in range(dx):
                self.x_obs.append(self.id_map[data_qubit + (x)])

        # Bad edges
        else:
            """raise ValueError(
                'Bad edge types; (North, South) must have the same edge types, and (East, West) must have the same edge types.')"""
            warnings.warn(
                "Cannot initialize observable qubits list. Either the North/South or East/West edge types do not match\n"
                "Explicitly declare the observables with <object>.x_obs:list[int] = [...] or <object>.z_obs:list[int] = [...]")

        self.obs_qubits = self.x_obs if self.basis == 'x' else self.z_obs

        """for qubit in self.qubits:
            patch_specific_neighbors = super().get_pqubit_neighbors(qubit.get_id())
            for neighbor in patch_specific_neighbors:
                qubit.add_neighbor(neighbor.get_id(),qubit.get_type() if qubit.get_type() != 'd' else None)"""

        self.q_map = {q.get_id(): q for q in self.qubits}

        # track edge qubits
        self.edges : dict[str:set[pqubit]] = {}
        for dir,edge_method in zip(
                [
                    "N",
                    "S",
                    "E",
                    "W"
                ],
                [
                    self.get_north_edge_qubits,
                    self.get_south_edge_qubits,
                    self.get_west_edge_qubits,
                    self.get_east_edge_qubits
                ]
        ):
            self.edges[dir] = set(edge_method())

    def get_qubit_coords_circuit(self):
        init_circuit = ''

        # INITIALIZE QUBIT COORDINATES
        for qubit in self.qubits:
            init_circuit += f"QUBIT_COORDS({qubit.get_pos_y()},{qubit.get_pos_x()}) {qubit.get_id()}\n"

        return init_circuit

    def get_init_circuit(self):

        init_circuit = ""

        # Determine whether to reset the qubit in the X or Z basis
        z_resets = []
        x_resets = []
        for qubit in self.qubits:
            if qubit.get_init_state() == '0' or qubit.get_init_state() == '1':
                z_resets.append(qubit)
            elif qubit.get_init_state() == '+' or qubit.get_init_state() == '-':
                x_resets.append(qubit)
            else:
                raise ValueError(f'Bad initial state on qubit {qubit.get_id()}')

        attr = 'after_reset_flip_probability'
        # Apply resets accordingly
        if len(z_resets) > 0:
            for qubit in z_resets:
                init_circuit += 'RZ '
                init_circuit += f'{qubit.get_id()} '
                init_circuit += '\n'
                init_circuit += f'X_ERROR({self.get_error_rate(qubit,attr):.15f}) '
                #for data_qubit in z_resets:
                init_circuit += f'{qubit.get_id()} '
                init_circuit += '\n'

        if len(x_resets) > 0:
            for qubit in x_resets:
                init_circuit += 'RX '
                init_circuit += f'{qubit.get_id()} '
                init_circuit += '\n'

                init_circuit += f'Z_ERROR({self.get_error_rate(qubit,attr):.15f}) '
                #for data_qubit in x_resets:
                init_circuit += f'{qubit.get_id()} '
                init_circuit += '\n'

        # Apply excitations to data qubits (if any)
        z_excitations = []
        x_excitations = []
        for qubit in self.qubits:
            if qubit.get_init_state() == '1':
                z_excitations.append(qubit)
            if qubit.get_init_state() == '-':
                x_excitations.append(qubit)

        # TODO - incorporate initialization error on excitations, note - unclear what's wrong here
        #if len(z_excitations) > 0 or len(x_excitations) > 0:
        #    raise NotImplementedError
        #"""
        if len(z_excitations) > 0:
            init_circuit += 'X '
            for qubit in z_excitations:
                init_circuit += f'{qubit.get_id()} '
            init_circuit += '\n'

        if len(x_excitations) > 0:
            init_circuit += 'X '
            for qubit in x_excitations:
                init_circuit += f'{qubit.get_id()} '
            init_circuit += '\n'
            init_circuit += 'H '
            for qubit in x_excitations:
                init_circuit += f'{qubit.get_id()} '
            init_circuit += '\n'
        #"""

        # TICK
        init_circuit += 'TICK\n'
        init_circuit += 'SHIFT_COORDS(0,0,1)\n'

        return init_circuit

    def construct_detector(self, parity_qid: int, num_comparisons: int, x:float=0, y:float=0, z:float=0) -> str:
        """
        Construct a detector for a given parity qubit.
        :param parity_qubit: Integer ID of the parity qubit
        :param num_comparisons: Number of prior measurements of the parity qubit to compare against
        :param x: X coordinate annoation
        :param y: Y coordinate annoation
        :param z: Z coordinate annoation
        """

        construction: str = f"DETECTOR({x},{y},{z}) "
        completed_comparisons: int = 0

        for m_index, m_id in enumerate(reversed(self.measurements)):

            if m_id == parity_qid:

                construction += f'rec[{-1 * (1 + m_index)}] '
                completed_comparisons += 1

                if completed_comparisons >= num_comparisons:
                    break

        return construction

    def get_quiescent_circuit(self):

        init_circuit = ''

        # Depolarize data qubits
        attr = 'before_round_data_depolarization'
        for data_qubit in self.data_qubits:
            error_rate = self.get_error_rate(data_qubit, attr)
            init_circuit  += f'DEPOLARIZE1({error_rate:.15f}) {data_qubit.get_id()}\n'

        # Construct stabilizer circuits
        init_circuit += self.get_stabilizer_circuit(1)

        # Apply detectors to relevant stabilizers
        for parity_qubit in self.parity_qubits:

            if parity_qubit.get_exclude_detectors():
                continue

            if parity_qubit.get_type() == self.basis:

                init_circuit += self.construct_detector(parity_qubit.get_id(), 1) + '\n'

        return init_circuit

    def get_rep_circuit(self, num_rounds):
        """
        Shouldn't be used as a public method. Will do repetitions of patches with shared stabilizers, but this circuit
        will not be a proper merged/surgeried circuit.
        :param without_repeat:
        :return:
        """

        init_circuit = ""

        data_qubits = [qubit for qubit in self.data_qubits]

        parity_qubits = [qubit for qubit in self.parity_qubits]

        # REPEAT BLOCK
        if num_rounds > 1:
            init_circuit += f"REPEAT {num_rounds} " + "{\n"

        # DATA IDLE
        attr = 'before_round_data_depolarization'
        for data_qubit in data_qubits:

            error_rate = self.get_error_rate(data_qubit,attr)
            init_circuit += f'DEPOLARIZE1({error_rate:.15f}) {data_qubit.get_id()}\n'

        # Construct stabilizer circuits (includes resetting parity qubits prior to CNOTs, performing CNOTs, and measuring
        #   after CNOTs. This method DOES track measurements internally for us in self.measurements.
        init_circuit += self.get_stabilizer_circuit(num_rounds)

        # SHIFT COORDS
        init_circuit += 'SHIFT_COORDS(0,0,1)\n'

        # APPLY DETECTORS BETWEEN ROUNDS
        measurements_reversed = self.measurements[::-1]
        for parity_qubit in parity_qubits:

            # Skip qubits who do not require detectors
            if parity_qubit.get_exclude_detectors():
                continue

            init_circuit += f"DETECTOR({parity_qubit.get_pos_y()},{parity_qubit.get_pos_x()},0) "

            # todo - make this an efficient reverse search...
            m_count = 0
            for m_index, m_id in enumerate(measurements_reversed):
                if m_id == parity_qubit.get_id():
                    init_circuit += (f'rec[{-1 * (m_index + 1)}] ')
                    m_count += 1

                if m_count >= 2:  # <-------Gets the 2nd latest measurement of the qubit
                    break

            # todo - why was this included?
            if m_count < 1:
                raise Warning(f'Couldn\'t find the prior round measurement for q{parity_qubit.get_id()}')

            init_circuit += '\n'

        # END REPEAT BLOCK
        if num_rounds > 1:
            init_circuit += "}\n"

        return init_circuit

    def get_terminating_circuit(self):
        init_circuit = ""

        # Measure all data qubits based on the multipatch/experiment basis
        error_string: str
        if self.basis == 'z':
            error_string = "X_ERROR"
        elif self.basis == 'x':
            error_string ="Z_ERROR"

        attr = 'before_measure_flip_probability'
        for data_qubit in self.data_qubits:
            init_circuit += error_string + f'({self.get_error_rate(data_qubit,attr):.15f}) {data_qubit.get_id()} '
            init_circuit += '\n'

        if self.basis == 'z':
            init_circuit += 'M '
        elif self.basis == 'x':
            init_circuit += 'MX '
        else:
            raise ValueError("Bad basis while creating termination circuit.")

        for data_qubit in self.data_qubits:
            init_circuit += f'{data_qubit.get_id()} '
            self.measurements.append(data_qubit.get_id())

        init_circuit += '\n'
        measurements_reversed = self.measurements[::-1]

        for parity_qubit in self.get_parity_qubits():

            if parity_qubit.get_exclude_detectors():
                continue

            if parity_qubit.get_type() == self.basis:
                init_circuit += f'DETECTOR({parity_qubit.get_pos_y()},{parity_qubit.get_pos_x()},1) '

                for neighbor_qubit in self.get_pqubit_neighbors(parity_qubit.get_pos()):

                    if neighbor_qubit == -1:  # <---- -1 is the indicator for no neighbor
                        continue

                    init_circuit += f'rec[{-1 * (1 + measurements_reversed.index(neighbor_qubit.get_id()))}] '

                try:
                    init_circuit += f'rec[{-1 * (1 + measurements_reversed.index(parity_qubit.get_id()))}]\n'
                except ValueError:
                    warnings.warn(f"Unable to locate measurement record for {parity_qubit.get_id()}")

        init_circuit += '\n'

        """if self.basis == 'x':
            init_circuit += 'OBSERVABLE_INCLUDE(0) '
            for qubit in self.x_obs:
                init_circuit += f'rec[{-1*(measurements_reversed.index(qubit.get_id())+1)}] '

        elif self.basis == 'z':
            init_circuit += 'OBSERVABLE_INCLUDE(0) '
            for qubit in self.z_obs:
                init_circuit += f'rec[{-1 * (measurements_reversed.index(qubit.get_id()) + 1)}] '

        else:
            raise ValueError("Observable unknown.")"""

        init_circuit += 'OBSERVABLE_INCLUDE(0) '
        for qubit in self.obs_qubits:
            init_circuit += f'rec[{-1 * (measurements_reversed.index(qubit.get_id()) + 1)}] '

        return init_circuit

    def get_full_circuit(self, num_rounds: int | None = None) -> str:
        """
        Construct the str corresponding to the full stim circuit that performs a memory experiment.
        :param num_rounds: Number of rounds to perform the memory experiment for. Defaults to ``self.rounds`` if unspecified.
        :return: Circuit string corresponding to the full memory experiment stim circuit.
        """

        if num_rounds is None:
            num_rounds = self.rounds

        self.measurements = []

        circ = ''

        circ += self.get_qubit_coords_circuit()
        circ += self.get_init_circuit()
        circ += self.get_quiescent_circuit()
        # for step_index in range(len(self.steps)):
        # self.next_step()
        for r in range(num_rounds):
            circ += self.get_rep_circuit(1)
        circ += self.get_terminating_circuit()
        return circ

    def get_west_edge_qubits(self):
        edge_qubits = []
        for qubit in self.qubits:
            if qubit.get_pos_x() <= 1:
                edge_qubits.append(qubit)

        return edge_qubits

    def get_east_edge_qubits(self):
        edge_qubits = []
        for qubit in self.qubits:
            if qubit.get_pos_x() >= (2 * self.dx + 1):
                edge_qubits.append(qubit)

        return edge_qubits

    def get_south_edge_qubits(self):
        edge_qubits = []
        for qubit in self.qubits:
            if qubit.get_pos_y() <= 1:
                edge_qubits.append(qubit)

        return edge_qubits

    def get_north_edge_qubits(self):
        edge_qubits = []
        for qubit in self.qubits:
            if qubit.get_pos_x() >= (2 * self.dx + 1):
                edge_qubits.append(qubit)

        return edge_qubits

    def get_west_edge_type(self):
        return self.edge_types['W']

    def get_east_edge_type(self):
        return self.edge_types['E']

    def get_north_edge_type(self):
        return self.edge_types['N']

    def get_south_edge_type(self):
        return self.edge_types['S']

    def get_metadata(self):
        return {
            'r': self.rounds,
            'b': self.basis,
            'dx': self.dx,
            'dy': self.dz,
            'psi_0': self.init_state,
            'p': self.uniform_error_rate,
            'origin': self.origin
        }

    def draw_patch(self, figsize=(8, 6), ax=None, draw_cx=True, show=True):
        from matplotlib import pyplot as plt
        import matplotlib.patches as mpatches
        import numpy as np

        ax_was_none = ax is None
        if ax is None:
            fig, ax = plt.subplots(1, 1, figsize=figsize)

        x = []
        y = []
        ids = []
        for qubit in self.qubits:
            x.append(qubit.get_pos_x())
            y.append(qubit.get_pos_y())
            ids.append(qubit.get_id())
            ax.text(
                qubit.get_pos_x()-0.05,
                qubit.get_pos_y()-0.05,
                qubit.get_id(),
                backgroundcolor='w',
                bbox=dict(facecolor='w', edgecolor='k',boxstyle='circle')
            )

        ax.scatter(x, y, label=self.id)

        for parity_qubit in self.parity_qubits:

            poly_points = []
            pos = parity_qubit.get_pos()
            try:
                poly_points.append(self.xy_map[(pos[0] - 1, pos[1] - 1)].get_pos()[::-1])
            except KeyError:
                pass
            try:
                poly_points.append(self.xy_map[(pos[0] - 1, pos[1] + 1)].get_pos()[::-1])
            except KeyError:
                pass
            try:
                poly_points.append(self.xy_map[(pos[0] + 1, pos[1] + 1)].get_pos()[::-1])
            except KeyError:
                pass
            try:
                poly_points.append(self.xy_map[(pos[0] + 1, pos[1] - 1)].get_pos()[::-1])
            except KeyError:
                pass

            if len(poly_points) < 3:
                poly_points.append(pos[::-1])
            poly = mpatches.Polygon(
                np.array(poly_points),
                facecolor='lightblue' if parity_qubit.get_type() == 'z' else 'coral',
                edgecolor='black',
                zorder=-2,
                linewidth=0.3,
                linestyle='-',
                alpha=0.6
            )
            ax.add_patch(poly)

            if draw_cx:
                neighbor_coordinates = []
                for neighbor in [qid for qid in self.get_pqubit_neighbors(parity_qubit.get_id())]:
                    if neighbor == -1:
                        continue
                    neighbor_coordinates.append(self.id_map[neighbor.get_id()].get_pos())
                for ni in range(1, len(neighbor_coordinates)):

                    prev_neighbor = neighbor_coordinates[ni - 1]

                    px_offset = -0.4
                    py_offset = -0.4
                    if prev_neighbor[0] < pos[0]:
                        py_offset = 0.4
                    if prev_neighbor[1] < pos[1]:
                        px_offset = 0.4

                    next_neighbor = neighbor_coordinates[ni]

                    nx_offset = -0.4
                    ny_offset = -0.4
                    if next_neighbor[0] < pos[0]:
                        ny_offset = 0.4
                    if next_neighbor[1] < pos[1]:
                        nx_offset = 0.4

                    ax.arrow(
                        prev_neighbor[1] + px_offset,
                        prev_neighbor[0] + py_offset,
                        (next_neighbor[1] + nx_offset) - (prev_neighbor[1] + px_offset),
                        (next_neighbor[0] + ny_offset) - (prev_neighbor[0] + py_offset),
                        facecolor='white',
                        edgecolor='black',
                        width=0.005,
                        head_width=0.15,
                        length_includes_head=True
                    )


        ax.grid()
        if show and ax_was_none:
            fig.show()
        if ax_was_none:
            return fig, ax

    def invert_stabilizers(self):
        for parity_qubit in self.parity_qubits:
            new_type = 'x' if parity_qubit.get_type() == 'z' else 'z'
            parity_qubit.set_type(new_type)

    def get_stabilizer_circuit(self, reps):
        """
        Returns the circuit that constructs all stabilizers in the topology.
        :return:
        """

        if reps > 1:
            raise NotImplementedError("Keep reps = 1 for now")

        init_circuit = ""

        attr = 'after_clifford_depolarization'

        # APPLY HADAMARDS TO X STABILIZERS
        for parity_qubit in self.parity_qubits:
            if parity_qubit.get_type() == 'x':
                init_circuit += f'H {parity_qubit.get_id()}\n'
                init_circuit += f'DEPOLARIZE1({self.get_error_rate(parity_qubit, attr):.15f}) {parity_qubit.get_id()}\n'

        # TICK
        init_circuit += 'TICK\n'

        cx_rounds: dict[int,list[tuple[int,int]]] = {}

        # PERFORM CNOTS TO "ACTIVATE" STABILIZERS
        for parity_qubit in self.parity_qubits:
            neighbors = [neighbor for neighbor in self.get_pqubit_neighbors(parity_qubit.get_id(),type_filter='d')]
            parity_qubit_id = parity_qubit.get_id()
            for ni, neighbor in enumerate(neighbors):

                # For skipping a round in the timings/neighbor orderings
                if neighbor == -1:
                    continue

                neighbor_id = neighbor.get_id()

                assert(neighbor is not None)

                if ni not in cx_rounds:
                    cx_rounds[ni] = []

                # x: parity targets data
                if parity_qubit.get_type() == 'x':
                    cx_rounds[ni].append((parity_qubit_id, neighbor_id))

                # z: data targets parity
                elif parity_qubit.get_type() == 'z':
                    cx_rounds[ni].append((neighbor_id, parity_qubit_id))

                else:
                    raise ValueError(f'Unknown CX type between qubits {parity_qubit_id} and {neighbor}')

        rounds = cx_rounds.keys()
        sorted_rounds = sorted(rounds)
        for round_num in sorted_rounds:

            cxs = cx_rounds[round_num]

            for cx in cxs:

                init_circuit += 'CX '

                init_circuit += f'{cx[0]} {cx[1]}\n'

                # apply 2-qubit error rate (if it exists)
                if cx in self.coupling_errors:
                    init_circuit += f'DEPOLARIZE2({self.coupling_errors[cx]:.15f}) {cx[0]} {cx[1]}\n'

        attr = 'after_clifford_depolarization'

        # UNDO HADAMARDS ON X STABILIZERS
        for parity_qubit in self.parity_qubits:
            if parity_qubit.get_type() == 'x':
                init_circuit += f'H {parity_qubit.get_id()}\n'
                init_circuit += f'DEPOLARIZE1({self.get_error_rate(parity_qubit, attr):.15f}) {parity_qubit.get_id()}\n'

        # TICK
        init_circuit += 'TICK\n'

        attr = 'before_measure_flip_probability'

        # APPLY BIT FLIP ERROR PRIOR TO MEASUREMENT OF STABILIZERS
        # MEASURE THE STABILIZERS
        for parity_qubit in self.parity_qubits:

            init_circuit += f'X_ERROR({self.get_error_rate(parity_qubit, attr):.15f}) {parity_qubit.get_id()}\n'
            init_circuit += f'MZ {parity_qubit.get_id()}\n'

            # TRACK MEASUREMENTS
            self.measurements.append(parity_qubit.get_id())

        init_circuit += '\n'

        attr = 'after_reset_flip_probability'

        # RESET THE Z STABILIZERS AND APPLY ERROR
        for parity_qubit in self.parity_qubits:
            #if parity_qubit.get_type() == 'z':
            init_circuit += f'RZ {parity_qubit.get_id()}\n'
            init_circuit += f'X_ERROR({self.get_error_rate(parity_qubit,attr):.15f}) {parity_qubit.get_id()}\n'

        # RESET THE X STABILIZERS AND APPLY ERROR
        # note - we could be cheeky and us RX to avoid using hadamards above
        """for parity_qubit in self.parity_qubits:
            if parity_qubit.get_type() == 'x':
                init_circuit += f'RX {parity_qubit.get_id()}\n'
                init_circuit += f'DEPOLARIZE1({self.get_error_rate(parity_qubit, attr)}) {parity_qubit.get_id()}\n'
        """

        return init_circuit

    def set_uniform_error_rate(self, uniform_error_rate: float | None) -> None:
        # TODO - make this error rate also a model...
        self.uniform_error_rate: float = 0.0 if uniform_error_rate is None else uniform_error_rate

        if uniform_error_rate is None:
            self.get_error_rate: callable[[float], float] = lambda q, e: getattr(q.get_error_model(), e) #if isinstance(q,pqubit) else

        else:
            self.get_error_rate: callable[[float], float] = lambda q, e: uniform_error_rate

    def set_obs(self, qids: list[int]):

        temp: list[pqubit] = []

        for qid in qids:

            # todo - add a way to check if a qid is a data qubit
            #if qid not in self.data_qubits:
            #    raise ValueError(f"q{qid} is not a data qubit.")

            temp.append(self.q_map[qid])

        self.obs_qubits = temp

    def get_obs(self):
        return self.obs_qubits


    def get_x_stab(self):
        return self.x_stab
    def get_z_stab(self):
        return self.z_stab

    def set_coupling_error_map(
            self,
            error_map: dict[tuple[int, int], float]
    ):
        """
        :param error_map: Dictionary of qid tuples to error rate (float) of the corresponding 2-qubit interactions
        :return: None
        """

        self.coupling_errors = error_map

    def get_coupling_error(self, q1: int, q2: int) -> float:
        return self.coupling_errors[(q1, q2)]

    def set_coupling_error(self, q1: int, q2: int, val: float) -> None:
        self.coupling_errors[(q1, q2)] = val