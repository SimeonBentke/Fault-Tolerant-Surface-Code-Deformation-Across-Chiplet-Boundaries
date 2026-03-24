from .pqubit import pqubit
class qubit_topology:

    x_stab_order: list[list[int]|int] = [
                                    [1,1],
                                    [-1,1],
                                    [1,-1],
                                    [-1,-1],
                                    #-1
                                    ]
    z_stab_order: list[list[int]|int] = [
                                    [1,1],
                                    [1,-1],
                                    [-1,1],
                                    [-1,-1],
                                    #-1
                                    ]

    stab_order_colors: list[str] = [
        'orange',
        'magenta',
        'cyan',
        'lime',
        'yellow',
        'beige',
        'salmon',
        'olive'
    ]

    qubits: list[pqubit] # todo - convert to set
    data_qubits: set[pqubit]
    parity_qubits: list[pqubit] # todo - convert to set
    obs_qubits: list[list[pqubit]] # todo - convert to set
    id: str = id
    pqubit_id_offset: int
    origin: tuple[float, float]
    measurements: list

    def __init__(self, id, pqubit_id_offset, origin):
        self.qubits: list[pqubit] = []
        self.data_qubits: set[pqubit] = set()
        self.parity_qubits: list[pqubit] = []
        self.obs_qubits: list[list[pqubit]] = []
        self.id: str = id
        self.pqubit_id_offset: int = pqubit_id_offset
        self.origin: tuple[float,float] = origin
        self.measurements: list = []

        # Init internal qubit/coordinate maps
        self._reset_qubit_maps()

    def get_num_qubits(self):
        return len(self.xy_map)

    def get_id(self):
        return self.id

    def get_max_id(self):
        return max(self.id_map)

    def get_min_id(self):
        return min(self.id_map)

    def _reset_qubit_maps(self):
        id_map, xy_map = self.get_qubit_maps()
        self.id_map: dict[int, pqubit] = id_map
        self.xy_map: dict[tuple[float, float], pqubit] = xy_map

    def delete_pqubit(self, qubit_id):
        """
        Deletes the pqubit corresponding to the qubit_id from the topology, and removes any reference of the pqubit from
        the neighboring pqubits.
        :param qubit_id:
        :return:
        """
        qubit = self.id_map[qubit_id]
        del self.id_map[qubit_id]
        del self.xy_map[qubit.get_pos()]
        self.qubits.remove(qubit)
        if qubit.get_type() != 'd':
            self.parity_qubits.remove(qubit)
        else:
            self.data_qubits.remove(qubit)

    def add_pqubit(self, id: int, pos: tuple[float,float], qtype: str, is_obs=False):
        new_qubit = pqubit(id, pos[0], pos[1], qtype)
        self.qubits.append(new_qubit)
        if qtype == 'd':
            self.data_qubits.add(new_qubit)
        elif qtype == 'x' or type == 'z':
            self.parity_qubits.append(new_qubit)
        else:
            raise ValueError(f'Bad qubit type: {type}.')

        self.xy_map[new_qubit.get_pos()] = new_qubit
        self.id_map[new_qubit.get_id()] = new_qubit

    def add_existing_pqubit(self, new_qubit: pqubit):
        self.qubits.append(new_qubit)
        if new_qubit.get_type() == 'd':
            self.data_qubits.add(new_qubit)
        elif new_qubit.get_type() == 'x' or new_qubit.get_type() == 'z':
            self.parity_qubits.append(new_qubit)
        else:
            raise ValueError(f'Bad qubit type: {new_qubit.get_type()}.')

        self.xy_map[new_qubit.get_pos()] = new_qubit
        self.id_map[new_qubit.get_id()] = new_qubit

    def replace_pqubit(self, old_qubit_id: int, new_qubit: pqubit, additional_neighbors: list[int] = None, cx_types: list[str] = None):

        if self.id_map[old_qubit_id].get_pos() != new_qubit.get_pos():
            raise ValueError('Cannot replace a pqubit if it does not have the same location as the new pqubit.')

        old_qubit = self.id_map[old_qubit_id]
        del self.id_map[old_qubit_id]
        del self.xy_map[old_qubit.get_pos()]
        self.qubits.remove(old_qubit)

        new_qubit_id = new_qubit.get_id()
        self.id_map[new_qubit_id] = new_qubit
        self.xy_map[old_qubit.get_pos()] = new_qubit
        self.qubits.append(new_qubit)

        if old_qubit.get_type() != 'd':
            self.parity_qubits.remove(old_qubit)
            self.parity_qubits.append(new_qubit)
        else:
            self.data_qubits.remove(old_qubit)
            self.data_qubits.add(new_qubit)

        try:
            self.obs_qubits.remove(old_qubit)
            self.obs_qubits.append(new_qubit)
        except ValueError:
            pass

        if additional_neighbors is not None:

            if len(additional_neighbors) != len(cx_types):
                raise ValueError('Number of additional neighbors and number of CNOT types does not match in qubit replacement.')

            for neighbor, cx_type in zip(additional_neighbors, cx_types):
                new_qubit.add_neighbor(neighbor, cx_type)

    def mirror_across_y(self):

        # Find max x pos
        max_x = self.origin[1]
        for qubit in self.qubits:
            if qubit.get_pos_x() >= max_x:
                max_x = qubit.get_pos_x()

        # Absolute diff between qubit x pos and max x is the new x pos
        for qubit in self.qubits:
            new_x = self.origin[1] + abs(qubit.get_pos_x()-max_x)
            qubit.set_pos_x(new_x)

        self._reset_qubit_maps()

    def mirror_across_x(self):

        # Find max x pos
        max_y = self.origin[0]
        for qubit in self.qubits:
            if qubit.get_pos_y() >= max_y:
                max_y = qubit.get_pos_y()

        # Absolute diff between qubit x pos and max x is the new x pos
        for qubit in self.qubits:
            new_y = self.origin[0] + abs(qubit.get_pos_y()-max_y)
            qubit.set_pos_y(new_y)

        self._reset_qubit_maps()

    def get_qubit_maps(self):
        id_map = {}
        xy_map = {}
        for qubit in self.qubits:
            id_map[qubit.get_id()] = qubit
            xy_map[qubit.get_pos()] = qubit
        return id_map, xy_map

    def get_qubit_from_id(self, qubit_id):
        return self.id_map[qubit_id]

    def get_qubit_from_pos(self, qubit_pos):
        return self.xy_map[qubit_pos]

    def get_qubit_coordinates(self, qubit_id):
        return self.id_map[qubit_id].get_pos()

    def get_qubit_id_from_coordinates(self, coordinates):
        return self.xy_map[coordinates].get_id()

    def get_qubits(self):
        return self.qubits

    def get_data_qubits(self):
        return self.data_qubits

    def get_parity_qubits(self):
        return self.parity_qubits

    def get_pqubit_neighbors(
            self,
            coords_or_id: tuple[float,float] | int,
            type_filter=None,
    ) -> list[pqubit | int]:

        """
        Returns neighbors in a sequence corresponding to the ordering of the CX performed for stabilizers.
        Data qubits default to Z-type stabilizer ordering for their neighbors.

        Fills in -1 as a mask value for either a missing neighbor or type filtering.

        Types: 'd', 'x', or 'z' (i.e. != 'd' implies it's a parity qubit)
        """

        if isinstance(coords_or_id, int):
            coords_or_id = self.id_map[coords_or_id].get_pos()

        qubit_exists: bool = coords_or_id in self.xy_map

        qtype: None | str = self.xy_map[coords_or_id].get_type() if qubit_exists else None

        # default to z-stab order
        ordering = self.x_stab_order if qtype == 'x' else self.z_stab_order
        if qubit_exists and self.xy_map[coords_or_id].custom_offsets:
            ordering = self.xy_map[coords_or_id].custom_offsets

        neighbors: list[pqubit | int] = []
        for offset in ordering:

            # append -1 on masking
            if offset == -1:
                neighbors.append(-1)
                continue

            try:

                # append pqubit neighbor on success/exists
                neighbor = self.xy_map[(coords_or_id[0] + offset[0], coords_or_id[1] + offset[1])]
                if type_filter is None or neighbor.get_type() == type_filter:
                    neighbors.append(neighbor)

                # append -1 on filtering
                elif neighbor.get_type() != type_filter:
                    neighbors.append(-1)

            # append -1 on missing
            except KeyError:
                neighbors.append(-1)

        return neighbors