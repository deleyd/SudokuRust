use rand::{Rng, SeedableRng};
use std::collections::HashMap;
fn print_board(board : &[[char; 13]; 13])
{
    for row in board {
        for &ch in row {
            print!("{}", ch); // Print each character without a newline
        }
        println!(); // Print a newline after each row
    }
}

fn main() {
    let line: &str = "+---+---+---+";
    let _middle: &str = "|...|...|...|";
    // Declares a 3x4 2D array of characters, initialized with 'X'
    let mut board: [[char; 13]; 13] = [['X'; 13]; 13];

    // Accessing and modifying elements
    let letters: Vec<char> = line.chars().collect();
    let line2: Vec<char> = _middle.chars().collect();
    board[0] = letters.clone().try_into().expect("REASON");
    board[1] = line2.clone().try_into().expect("REASON");
    board[2] = line2.clone().try_into().expect("REASON");
    board[3] = line2.clone().try_into().expect("REASON");
    board[4] = letters.clone().try_into().expect("REASON");
    board[5] = line2.clone().try_into().expect("REASON");
    board[6] = line2.clone().try_into().expect("REASON");
    board[7] = line2.clone().try_into().expect("REASON");
    board[8] = letters.clone().try_into().expect("REASON");
    board[9] = line2.clone().try_into().expect("REASON");
    board[10] = line2.clone().try_into().expect("REASON");
    board[11] = line2.clone().try_into().expect("REASON");
    board[12] = letters.clone().try_into().expect("REASON");
    // Iterating and printing
    for row in &board {
        for &c in row {
            print!("{}", c);
        }
        println!();
    }
    println!("XYZZY");

    // Construct board to be solved
    let mut rng = rand_chacha::ChaCha8Rng::seed_from_u64(10);
    println!("Random u32: {}", rng.random::<u32>());

    // Top element is current state of the board
    //Stack<int[]> state_stack = new Stack<int[]>();
    let mut state_stack: Vec<[u32; 9 * 9]> = Vec::new(); // Explicitly typed for clarity

    // Top elements are (row, col) of cell which has been modified compared to previous state
    //Stack<int> rowIndexStack = new Stack<int>();
    //Stack<int> colIndexStack = new Stack<int>();
    let mut row_index_stack: Vec<usize> = Vec::new(); // Explicitly typed for clarity
    let mut col_index_stack: Vec<usize> = Vec::new(); // Explicitly typed for clarity

    // Top element indicates candidate digits (those with False) for (row, col)
    //Stack<bool[]> usedDigitsStack = new Stack<bool[]>();
    let mut used_digits_stack: Vec<[bool; 9]> = Vec::new();

    // Top element is the value that was set on (row, col)
    //Stack<int> lastDigitStack = new Stack<int>();
    let mut last_digit_stack: Vec<u32> = Vec::new();

    // Indicates operation to perform next
    // - expand - finds next empty cell and puts new state on stacks
    // - move - finds next candidate number at current pos and applies it to current state
    // - collapse - pops current state from stack as it did not yield a solution
    let mut command: &str = "expand";
    println!("state_stack.len() ={}", state_stack.len());
    while state_stack.len() <= 9 * 9
    {
        println!("before if");
        if command == "expand"
        {
            println!("in if");
            let mut current_state: [u32; 81] = [0; 81];
            if state_stack.len() > 0
            {
                // source array is state_stack.last(), destination is current_state, length is 81
                //Array.Copy(state_stack.last(), current_state, current_state.len());
                let opt_owned = state_stack.last().cloned();
                current_state = opt_owned.unwrap();
            }

            let mut best_row: usize = 999;
            let mut best_col: usize = 999;
            let mut best_used_digits: [bool; 9] = [false; 9];
            let mut best_candidates_count: i32 = -1;
            let mut best_random_value: i32 = -1;
            let mut contains_unsolvable_cells: bool = false;
            for index in 0..81
            {
                if current_state[index] == 0
                {
                    let row: usize = index / 9;
                    let col: usize = index % 9;
                    let block_row: usize = row / 3;
                    let block_col: usize = col / 3;

                    let mut is_digit_used: [bool; 9] = [false; 9];

                    for i in 0..9
                    {
                        let row_digit = current_state[9 * i + col as usize];
                        if row_digit > 0
                        {
                            is_digit_used[row_digit as usize - 1] = true;
                        }

                        let col_digit = current_state[9 * row as usize + i];
                        if col_digit > 0
                        {
                            is_digit_used[col_digit as usize - 1] = true;
                        }

                        let block_digit = current_state[(block_row as usize * 3 + i / 3) * 9 + (block_col as usize * 3 + i % 3)];
                        if block_digit > 0
                        {
                            is_digit_used[block_digit as usize - 1] = true;
                        }
                        let candidates_count: i32 = is_digit_used.iter() // Get an iterator over the elements
                            .filter(|&&value| !value) // Filter for false values
                            .count() as i32; // Count the remaining elements
                        //let candidates_count = is_digit_used.Where(used => !used).Count();

                        if candidates_count == 0
                        {
                            contains_unsolvable_cells = true;
                            break;
                        }

                        let random_value = rng.random::<i32>();

                        if best_candidates_count < 0 ||
                            candidates_count < best_candidates_count ||
                            (candidates_count == best_candidates_count && random_value < best_random_value)
                        {
                            best_row = row;
                            best_col = col;
                            best_used_digits = is_digit_used;
                            best_candidates_count = candidates_count;
                            best_random_value = random_value;
                        }
                    } // for (i = 0..8)
                }

                if !contains_unsolvable_cells
                {
                    state_stack.push(current_state);
                    row_index_stack.push(best_row);
                    col_index_stack.push(best_col);
                    used_digits_stack.push(best_used_digits);
                    last_digit_stack.push(0); // No digit was tried at this position
                }

                // Always try to move after expand
                command = "move";
            }
        } // if (command == "expand")
        else if command == "collapse"
        {
            state_stack.pop();
            row_index_stack.pop();
            col_index_stack.pop();
            used_digits_stack.pop();
            last_digit_stack.pop();

            command = "move";   // Always try to move after collapse
        } else if command == "move"
        {
            let row_to_move = row_index_stack.last().unwrap();  // panic if empty which it should never be
            let col_to_move = col_index_stack.last().unwrap();
            let digit_to_move: u32 = *last_digit_stack.last().unwrap();

            let row_to_write = row_to_move + row_to_move / 3 + 1;
            let col_to_write = col_to_move + col_to_move / 3 + 1;

            let mut used_digits = used_digits_stack.last().unwrap().clone();
            let mut current_state = state_stack.last().unwrap().clone();
            let current_state_index = 9 * row_to_move + col_to_move;

            let mut moved_to_digit = digit_to_move + 1;
            while moved_to_digit <= 9 && used_digits[moved_to_digit as usize - 1]
            {
                moved_to_digit += 1;
            }

            if digit_to_move > 0
            {
                used_digits[digit_to_move as usize - 1] = false;
                current_state[current_state_index] = 0;
                board[row_to_write][col_to_write] = '.';
            }

            if moved_to_digit <= 9
            {
                last_digit_stack.push(moved_to_digit);
                used_digits[moved_to_digit as usize - 1] = true;
                current_state[current_state_index] = moved_to_digit;
                // board[row_to_write][col_to_write] = /*(char)*/
                let _m = '0' as u32 + moved_to_digit;

                // Next possible digit was found at current position
                // Next step will be to expand the state
                command = "expand";
            } else {
                // No viable candidate was found at current position - pop it in the next iteration
                last_digit_stack.push(0);
                command = "collapse";
            }
        } // if (command == "move")
    }
    println!();
    println!("Final look of the solved board:");
    print_board(&board);
    /*
    let transformed_data: Vec<_> = board.iter() // Iterate over outer Vec<Vec<char>>
        .map(|inner_vec| { // For each inner Vec<char>
            inner_vec.iter() // Iterate over inner Vec<char>
                .map(|&c| c.to_string().to_uppercase()) // Transform each char to uppercase String
                .collect::<Vec<_>>() // Collect transformed characters into a new Vec<String>
        })
        .collect(); // Collect all inner Vec<String> into a new Vec<Vec<String>>

    println!("{:?}", transformed_data); // Output: [["A", "B"], ["C", "D", "E"]]

    //println!(board);
     */
    //#endregion

    //#region Generate inital board from the completely solved one
    // Board is solved at this point.
    // Now pick subset of digits as the starting position.
    let mut remaining_digits = 30;
    let max_removed_per_block = 6;
    let mut removed_per_block: [[u32; 3]; 3] = [[0; 3]; 3];
    //int[] positions = Enumerable.Range(0, 9 * 9).ToArray();
    let mut positions: [usize; 9 * 9] = std::array::from_fn(|i| i as usize);
    let state = state_stack.last().unwrap().cloned();

    let final_state = state.cloned(); // new int[state.len()];
    //Array.Copy(state, final_state, final_state.len());

    let mut removed_pos = 0;
    while removed_pos < 9 * 9 - remaining_digits
    {
        let cur_remaining_digits = positions.len() - removed_pos;
        let index_to_pick = removed_pos + rng.random_range(0..cur_remaining_digits);

        let row: usize = positions[index_to_pick] / 9;
        let col: usize = positions[index_to_pick] % 9;

        let block_row_to_remove = row / 3;
        let block_col_to_remove = col / 3;

        if removed_per_block[block_row_to_remove][block_col_to_remove] >= max_removed_per_block
        {
            continue;
        }

        removed_per_block[block_row_to_remove][block_col_to_remove] += 1;

        let temp = positions[removed_pos];
        positions[removed_pos] = positions[index_to_pick];
        positions[index_to_pick] = temp;

        let row_to_write: usize = row + row / 3 + 1;
        let col_to_write: usize = col + col / 3 + 1;

        board[row_to_write][col_to_write] = '.';

        let state_index: usize = 9 * row + col;
        state[state_index] = 0;

        removed_pos += 1;
    }

    println!();
    println!("Starting look of the board to solve:");
    print_board(&board);
    //#endregion

    //#region Prepare lookup structures that will be used in further execution
    println!();
    let s = "=".repeat(80);
    println!(s);
    println!();

    let mut mask_to_ones_count : HashMap<usize,usize> = HashMap::new();
    mask_to_ones_count.insert(0, 0);
    for i in 1..(1 << 9)
    {
        let smaller : usize = i >> 1;
        let increment : usize = i & 1;
        let usize_value = mask_to_ones_count.get(&smaller)
            .and_then(|opt_val| opt_val.map(|val| val as usize))
            .unwrap_or(0);
        mask_to_ones_count.insert(i, usize_value + increment);
    }

    //Dictionary < int, int > single_bit_to_index = new
    let mut single_bit_to_index: HashMap<usize,usize> = HashMap::new();

    for i in 0..9
    {
        single_bit_to_index.insert(1 << i, i);
    }

    let all_ones = (1 << 9) - 1;
    //#endregion

    let mut change_made : bool = true;
    while (change_made)
    {
        change_made = false;

        //#region Calculate candidates for current state of the board
        let mut candidate_masks : [u32; 81] = [0; u32];

        for i in 0..state.len()
        {
            if (state[i] == 0)
            {
                let row = i / 9;
                let col = i % 9;
                let block_row = row / 3;
                let block_col = col / 3;

                let mut colliding_numbers = 0;
                for j in 0..9
                {
                    let row_sibling_index = 9 * row + j;
                    let col_sibling_index = 9 * j + col;
                    let block_sibling_index = 9 * (block_row * 3 + j / 3) + block_col * 3 + j % 3;

                    let row_sibling_mask = 1 << (state[row_sibling_index] - 1);
                    let col_sibling_mask = 1 << (state[col_sibling_index] - 1);
                    let block_sibling_mask = 1 << (state[block_sibling_index] - 1);

                    colliding_numbers = colliding_numbers | row_sibling_mask | col_sibling_mask | block_sibling_mask;
                }

                candidate_masks[i] = all_ones & !colliding_numbers;
            }
        }
        //#endregion
/*
        //#region Build a collection (named cellGroups) which maps cell indices into distinct groups (rows/columns/blocks)
        let rows_indices = state
            .Select((value, index) => new
            {
                Discriminator = index / 9,
                Description = $ "row #{index / 9 + 1}",
                Index = index,
                Row = index / 9,
                Column = index % 9
            })
            .GroupBy(tuple => tuple.Discriminator);

        let column_indices = state
            .Select((value, index) => new
            {
                Discriminator = 9 + index % 9,
                Description = $ "column #{index % 9 + 1}",
                Index = index,
                Row = index / 9,
                Column = index % 9
            })
            .GroupBy(tuple => tuple.Discriminator);

        let block_indices = state
            .Select((value, index) => new
            {
                Row = index / 9,
                Column = index % 9,
                Index = index
            })
            .Select(tuple => new
            {
                Discriminator = 18 + 3 * (tuple.Row / 3) + tuple.Column / 3,
                Description = $ "block ({tuple.Row / 3 + 1}, {tuple.Column / 3 + 1})",
                Index = tuple.Index,
                Row = tuple.Row,
                Column = tuple.Column
            })
            .GroupBy(tuple => tuple.Discriminator);

        let cell_groups = rows_indices.Concat(column_indices).Concat(block_indices).ToList();
        //#endregion
*/
        let mut step_change_made : bool = true;
        while step_change_made
        {
            step_change_made = false;

            //#region Pick cells with only one candidate left

            let single_candidate_indices =
                candidate_masks
                    .Select((mask, index) => new
                    {
                        CandidatesCount = maskToOnesCount: mask_to_ones_count[mask],
                        Index = index
                    })
                    .Where(tuple => tuple.CandidatesCount == 1)
                    .Select(tuple => tuple.Index)
                    .ToArray();

            if single_candidate_indices.len() > 0
            {
                let pick_single_candidate_index = rng.Next(single_candidate_indices.Length);
                let single_candidate_index = single_candidate_indices[pick_single_candidate_index];
                let candidate_mask = candidate_masks[single_candidate_index];
                let candidate = single_bit_to_index[candidate_mask];

                let row = single_candidate_index / 9;
                let col = single_candidate_index % 9;

                let row_to_write = row + row / 3 + 1;
                let col_to_write = col + col / 3 + 1;

                state[single_candidate_index] = candidate + 1;
                board[row_to_write][col_to_write] = (char)('1' + candidate);
                candidate_masks[single_candidate_index] = 0;
                change_made = true;

                println!("({0}, {1}) can only contain {2}.", row + 1, col + 1, candidate + 1);
            }

            //#endregion*

            //#region Try to find a number which can only appear in one place in a row/column/block

            if !change_made
            {
                let mut group_descriptions : Vec<String> = Vec::new();
                let mut candidate_row_indices : Vec<String> = Vec::new();
                let mut candidate_col_indices : Vec<String> = Vec::new();
                let mut candidates : Vec<String> = Vec::new();

                for digit in 1..=9
                {
                    let mask = 1 << (digit - 1);
                    for cell_group in 0..9
                    {
                        let mut row_number_count = 0;
                        let mut index_in_row = 0;

                        let mut col_number_count = 0;
                        let mut index_in_col = 0;

                        let mut block_number_count = 0;
                        let mut index_in_block = 0;

                        for index_in_group in 0..9
                        {
                            let row_state_index = 9 * cell_group + index_in_group;
                            let col_state_index = 9 * index_in_group + cell_group;
                            let block_row_index = (cell_group / 3) * 3 + index_in_group / 3;
                            let block_col_index = (cell_group % 3) * 3 + index_in_group % 3;
                            let block_state_index = block_row_index * 9 + block_col_index;

                            if (candidate_masks[row_state_index] & mask) != 0
                            {
                                row_number_count += 1;
                                index_in_row = index_in_group;
                            }

                            if (candidate_masks[col_state_index] & mask) != 0
                            {
                                col_number_count += 1;
                                index_in_col = index_in_group;
                            }

                            if (candidate_masks[block_state_index] & mask) != 0
                            {
                                block_number_count += 1;
                                index_in_block = index_in_group;
                            }
                        }

                        if row_number_count == 1
                        {
                            group_descriptions.push(format!("Row #{}", cell_group + 1));
                            candidate_row_indices.push(cell_group);
                            candidate_col_indices.push(index_in_row);
                            candidates.push(digit);
                        }

                        if col_number_count == 1
                        {
                            group_descriptions.Add($ "Column #{cell_group + 1}");
                            candidate_row_indices.Add(index_in_col);
                            candidate_col_indices.Add(cell_group);
                            candidates.Add(digit);
                        }

                        if (block_number_count == 1)
                        {
                            let block_row = cell_group / 3;
                            let block_col = cell_group % 3;

                            group_descriptions.Add($ "Block ({block_row + 1}, {block_col + 1})");
                            candidate_row_indices.Add(block_row * 3 + index_in_block / 3);
                            candidate_col_indices.Add(block_col * 3 + index_in_block % 3);
                            candidates.Add(digit);
                        }
                    } // for (cell_group = 0..8)
                } // for (digit = 1..9)

                if (candidates.Count > 0)
                {
                    let index = rng.Next(candidates.Count);
                    string
                    description = group_descriptions.ElementAt(index);
                    let row = candidate_row_indices.ElementAt(index);
                    let col = candidate_col_indices.ElementAt(index);
                    let digit = candidates.ElementAt(index);
                    let row_to_write = row + row / 3 + 1;
                    let col_to_write = col + col / 3 + 1;

                    string
                    message = $ "{description} can contain {digit} only at ({row + 1}, {col + 1}).";

                    let state_index = 9 * row + col;
                    state[state_index] = digit;
                    candidate_masks[state_index] = 0;
                    board[row_to_write][col_to_write] = (char)('0' + digit);

                    change_made = true;

                    println!(message);
                }
            }

            //#endregion

            //#region Try to find pairs of digits in the same row/column/block and remove them from other colliding cells
            if (!change_made)
            {
                IEnumerable < int > two_digit_masks =
                    candidate_masks.Where(mask => mask_to_ones_count[mask] == 2).Distinct().ToList();

                var
                groups =
                    two_digit_masks
                        .SelectMany(mask =>
                                    cell_groups
                                        .Where(group => group.Count(tuple => candidate_masks[tuple.Index] == mask) == 2)
                                        .Where(group => group.Any(tuple => candidate_masks[tuple.Index] != mask && (candidate_masks[tuple.Index] & mask) > 0))
                                        .Select(group => new
                                        {
                                            Mask = mask,
                                            Discriminator = group.Key,
                                            Description = group.First().Description,
                                            Cells = group
                                        }))
                        .ToList();

                if (groups.Any())
                {
                    foreach(var group in groups)
                    {
                        var
                        cells =
                            group.Cells
                                .Where(
                                    cell =>
                                    candidate_masks[cell.Index] != group.Mask &&
                                        (candidate_masks[cell.Index] & group.Mask) > 0)
                                .ToList();

                        var
                        mask_cells =
                            group.Cells
                                .Where(cell => candidate_masks[cell.Index] == group.Mask)
                                .ToArray();


                        if (cells.Any())
                        {
                            let upper = 0;
                            let lower = 0;
                            let temp = group.Mask;

                            let value = 1;
                            while (temp > 0)
                            {
                                if ((temp & 1) > 0)
                                {
                                    lower = upper;
                                    upper = value;
                                }
                                temp = temp >> 1;
                                value += 1;
                            }

                            println!(
                                $"Values {lower} and {upper} in {group.Description} are in cells ({mask_cells[0].Row + 1}, {mask_cells[0].Column + 1}) and ({mask_cells[1].Row + 1}, {mask_cells[1].Column + 1}).");

                            foreach(var cell in cells)
                            {
                                let mask_to_remove = candidate_masks[cell.Index] & group.Mask;
                                List < int > values_to_remove = new
                                List < int > ();
                                let cur_value = 1;
                                while (mask_to_remove > 0)
                                {
                                    if ((mask_to_remove & 1) > 0)
                                    {
                                        values_to_remove.Add(cur_value);
                                    }
                                    mask_to_remove = mask_to_remove >> 1;
                                    cur_value += 1;
                                }

                                string
                                valuesReport = string.Join(", ", values_to_remove.ToArray());
                                println!($"{valuesReport} cannot appear in ({cell.Row + 1}, {cell.Column + 1}).");

                                candidate_masks[cell.Index] &= ~group.Mask;
                                step_change_made = true;
                            }
                        }
                    }
                }
            }
            //#endregion

            //#region Try to find groups of digits of size N which only appear in N cells within row/column/block
            // When a set of N digits only appears in N cells within row/column/block, then no other digit can appear in the same set of cells
            // All other candidates can then be removed from those cells

            if (!change_made && !step_change_made)
            {
                IEnumerable < int > masks =
                    mask_to_ones_count
                        .Where(tuple => tuple.Value > 1)
                        .Select(tuple => tuple.Key).ToList();

                var
                groupsWithNMasks =
                    masks
                        .SelectMany(mask =>
                                    cell_groups
                                        .Where(group => group.All(cell => state[cell.Index] == 0 || (mask & (1 << (state[cell.Index] - 1))) == 0))
                                        .Select(group => new
                                        {
                                            Mask = mask,
                                            Description = group.First().Description,
                                            Cells = group,
                                            CellsWithMask =
                                            group.Where(cell => state[cell.Index] == 0 & & (candidate_masks[cell.Index] & mask) != 0).ToList(),
                                            CleanableCellsCount =
                                            group.Count(
                                            cell => state[cell.Index] == 0 & &
                                            (candidate_masks[cell.Index] & mask) != 0 & &
                                            (candidate_masks[cell.Index] & ~mask) != 0)
                                        }))
                        .Where(group => group.CellsWithMask.Count() == mask_to_ones_count[group.Mask])
                        .ToList();

                foreach(var group_with_n_masks in groupsWithNMasks)
                {
                    let mask = group_with_n_masks.Mask;

                    if (group_with_n_masks.Cells
                        .Any(cell =>
                             (candidate_masks[cell.Index] & mask) != 0 &&
                                 (candidate_masks[cell.Index] & ~mask) != 0))
                    {
                        StringBuilder
                        message = new
                        StringBuilder();
                        message.Append($ "In {group_with_n_masks.Description} values ");

                        string
                        separator = string.Empty;
                        let temp = mask;
                        let cur_value = 1;
                        while (temp > 0)
                        {
                            if ((temp & 1) > 0)
                            {
                                message.Append($ "{separator}{cur_value}");
                                separator = ", ";
                            }
                            temp = temp >> 1;
                            cur_value += 1;
                        }

                        message.Append(" appear only in cells");
                        foreach(var cell in group_with_n_masks.CellsWithMask)
                        {
                            message.Append($ " ({cell.Row + 1}, {cell.Column + 1})");
                        }

                        message.Append(" and other values cannot appear in those cells.");

                        println!(message.ToString());
                    }

                    foreach(var cell in group_with_n_masks.CellsWithMask)
                    {
                        let mask_to_clear = candidate_masks[cell.Index] & ~group_with_n_masks.Mask;
                        if (mask_to_clear == 0)
                        {
                            continue;
                        }

                        candidate_masks[cell.Index] &= group_with_n_masks.Mask;
                        step_change_made = true;

                        let mut value_to_clear = 1;

                        string
                        separator = string.Empty;
                        let mut message: String;

                        while mask_to_clear > 0
                        {
                            if (mask_to_clear) & 1 > 0
                            {
                                message.Append($ "{separator}{value_to_clear}");
                                separator = ", ";
                            }
                            mask_to_clear = mask_to_clear >> 1;
                            value_to_clear += 1;
                        }

                        message.Append($ " cannot appear in cell ({cell.Row + 1}, {cell.Column + 1}).");
                        println!(message.ToString());
                    }
                }
            }

            //#endregion
        }

        //#region Final attempt - look if the board has multiple solutions
        if (!change_made)
        {
            // This is the last chance to do something in this iteration:
            // If this attempt fails, board will not be entirely solved.

            // Try to see if there are pairs of values that can be exchanged arbitrarily
            // This happens when board has more than one valid solution

            //Queue<int> candidate_index1 = new Queue<int>();
            //Queue<int> candidate_index2 = new Queue<int>();
            //Queue<int> candidate_digit1 = new Queue<int>();
            //Queue<int> candidate_digit2 = new Queue<int>();
            let mut candidate_index1: VecDeque<i32> = VecDeque::new();
            let mut candidate_index2: VecDeque<i32> = VecDeque::new();
            let mut candidate_digit1: VecDeque<i32> = VecDeque::new();
            let mut candidate_digit2: VecDeque<i32> = VecDeque::new();

            for (int i = 0;
            i < candidate_masks.Length - 1;
            i + +)
            {
                if (mask_to_ones_count[candidate_masks[i]] == 2)
                {
                    let row = i / 9;
                    let col = i % 9;
                    let block_index = 3 * (row / 3) + col / 3;

                    let temp = candidate_masks[i];
                    let lower = 0;
                    let upper = 0;
                    for (int digit = 1;
                    temp > 0;
                    digit + +)
                    {
                        if ((temp & 1) != 0)
                        {
                            lower = upper;
                            upper = digit;
                        }
                        temp = temp >> 1;
                    }

                    for (int j = i + 1;
                    j < candidate_masks.Length;
                    j + +)
                    {
                        if (candidate_masks[j] == candidate_masks[i])
                        {
                            let row1 = j / 9;
                            let col1 = j % 9;
                            let block_index1 = 3 * (row1 / 3) + col1 / 3;

                            if (row == row1 || col == col1 || block_index == block_index1)
                            {
                                candidate_index1.Enqueue(i);
                                candidate_index2.Enqueue(j);
                                candidate_digit1.Enqueue(lower);
                                candidate_digit2.Enqueue(upper);
                            }
                        }
                    }
                }
            }

            // At this point we have the lists with pairs of cells that might pick one of two digits each
            // Now we have to check whether that is really true - does the board have two solutions?

            //List<int> state_index1 = new List<int>();
            //List<int> state_index2 = new List<int>();
            //List<int> value1 = new List<int>();
            //List<int> value2 = new List<int>();
            let mut state_index1: Vec<usize> = Vec::new();
            let mut state_index2: Vec<usize> = Vec::new();
            let mut value1: Vec<usize> = Vec::new();
            let mut value2: Vec<usize> = Vec::new();

            while (candidate_index1.Any())
            {
                let index1 = candidate_index1.Dequeue();
                let index2 = candidate_index2.Dequeue();
                let digit1 = candidate_digit1.Dequeue();
                let digit2 = candidate_digit2.Dequeue();

                int
                []
                alternate_state = new
                int[final_state.len()];
                Array.Copy(state, alternate_state, alternate_state.Length);

                if (final_state[index1] == digit1)
                {
                    alternate_state[index1] = digit2;
                    alternate_state[index2] = digit1;
                } else {
                    alternate_state[index1] = digit1;
                    alternate_state[index2] = digit2;
                }

                // What follows below is a complete copy-paste of the solver which appears at the beginning of this method
                // However, the algorithm couldn't be applied directly and it had to be modified.
                // Implementation below assumes that the board might not have a solution.
                state_stack = new
                Stack < int
                [] > ();
                row_index_stack = new
                Stack < int > ();
                col_index_stack = new
                Stack < int > ();
                used_digits_stack = new
                Stack < bool
                [] > ();
                last_digit_stack = new
                Stack < int > ();

                command = "expand";
                while (command != "complete" && command != "fail")
                {
                    if (command == "expand")
                    {
                        int
                        []
                        current_state = new
                        int[9 * 9];

                        if (state_stack.Any())
                        {
                            Array.Copy(state_stack.Peek(), current_state, current_state.Length);
                        } else {
                            Array.Copy(alternate_state, current_state, current_state.Length);
                        }

                        let best_row = -1;
                        let best_col = -1;
                        bool
                        []
                        best_used_digits = null;
                        let best_candidates_count = -1;
                        let best_random_value = -1;
                        bool
                        contains_unsolvable_cells = false;

                        for (int index = 0;
                        index < current_state.Length;
                        index + +)
                        if (current_state[index] == 0)
                        {
                            int
                            row = index / 9;
                            let col = index % 9;
                            let block_row = row / 3;
                            let block_col = col / 3;

                            bool
                            []
                            is_digit_used = new
                            bool[9];

                            for (int i = 0;
                            i < 9;
                            i + +)
                            {
                                let row_digit = current_state[9 * i + col];
                                if (row_digit > 0)
                                is_digit_used[row_digit - 1] = true;

                                let col_digit = current_state[9 * row + i];
                                if (col_digit > 0)
                                is_digit_used[col_digit - 1] = true;

                                let block_digit = current_state[(block_row * 3 + i / 3) * 9 + (block_col * 3 + i % 3)];
                                if (block_digit > 0)
                                is_digit_used[block_digit - 1] = true;
                            } // for (i = 0..8)

                            let candidates_count = is_digit_used.Where(used => !used).Count();

                            if (candidates_count == 0)
                            {
                                contains_unsolvable_cells = true;
                                break;
                            }

                            let random_value = rng.Next();

                            if (best_candidates_count < 0 ||
                                candidates_count < best_candidates_count ||
                                (candidates_count == best_candidates_count && random_value < best_random_value))
                            {
                                best_row = row;
                                best_col = col;
                                best_used_digits = is_digit_used;
                                best_candidates_count = candidates_count;
                                best_random_value = random_value;
                            }
                        } // for (index = 0..81)

                        if (!contains_unsolvable_cells)
                        {
                            state_stack.Push(current_state);
                            row_index_stack.Push(best_row);
                            col_index_stack.Push(best_col);
                            used_digits_stack.Push(best_used_digits);
                            last_digit_stack.Push(0); // No digit was tried at this position
                        }

                        // Always try to move after expand
                        command = "move";
                    } // if (command == "expand")
                    else if (command == "collapse")
                    {
                        state_stack.Pop();
                        row_index_stack.Pop();
                        col_index_stack.Pop();
                        used_digits_stack.Pop();
                        last_digit_stack.Pop();

                        if (state_stack.Any())
                        command = "move"; // Always try to move after collapse
                        else
                        command = "fail";
                    } else if (command == "move")
                    {
                        let row_to_move: usize = row_index_stack.last().cloned();
                        let col_to_move: usize = col_index_stack.last().cloned();
                        let digit_to_move = last_digit_stack.pop();

                        let row_to_write: usize = row_to_move + row_to_move / 3 + 1;
                        let col_to_write: usize = col_to_move + col_to_move / 3 + 1;

                        bool
                        []
                        used_digits = used_digits_stack.last().cloned();
                        let []
                        current_state = state_stack.last().cloned();
                        let current_state_index: usize = 9 * row_to_move + col_to_move;

                        let moved_to_digit = digit_to_move + 1;
                        while (moved_to_digit <= 9 && used_digits[moved_to_digit - 1])
                        moved_to_digit += 1;

                        if (digit_to_move > 0)
                        {
                            used_digits[digit_to_move - 1] = false;
                            current_state[current_state_index] = 0;
                            board[row_to_write][col_to_write] = '.';
                        }

                        if (moved_to_digit <= 9)
                        {
                            last_digit_stack.push(moved_to_digit);
                            used_digits[moved_to_digit - 1] = true;
                            current_state[current_state_index] = moved_to_digit;
                            board[row_to_write][col_to_write] = / *(char) * / ('0' + moved_to_digit);

                            if (current_state.Any(digit => digit == 0))
                            command = "expand";
                            else
                            command = "complete";
                        } else {
                            // No viable candidate was found at current position - pop it in the next iteration
                            last_digit_stack.push(0);
                            command = "collapse";
                        }
                    } // if (command == "move")

                } // while (command != "complete" && command != "fail")

                if (command == "complete")
                {   // Board was solved successfully even with two digits swapped
                    state_index1.Add(index1);
                    state_index2.Add(index2);
                    value1.Add(digit1);
                    value2.Add(digit2);
                }
            } // while (candidate_index1.Any())

            if (state_index1.Any())
            {
                let pos = rng.Next(state_index1.len());
                let index1 = state_index1.ElementAt(pos);
                let index2 = state_index2.ElementAt(pos);
                let digit1 = value1.ElementAt(pos);
                let digit2 = value2.ElementAt(pos);
                let row1 = index1 / 9;
                let col1 = index1 % 9;
                let row2 = index2 / 9;
                let col2 = index2 % 9;

                let description: string;

                if (index1 / 9 == index2 / 9)
                {
                    description = $ "row #{index1 / 9 + 1}";
                } else if (index1 % 9 == index2 % 9)
                {
                    description = $ "column #{index1 % 9 + 1}";
                } else {
                    description = $ "block ({row1 / 3 + 1}, {col1 / 3 + 1})";
                }

                state[index1] = final_state[index1];
                state[index2] = final_state[index2];
                candidate_masks[index1] = 0;
                candidate_masks[index2] = 0;
                change_made = true;

                for (int i = 0;
                i < state.Length;
                i + +)
                {
                    let temp_row = i / 9;
                    let temp_col = i % 9;
                    let row_to_write = temp_row + temp_row / 3 + 1;
                    let col_to_write = temp_col + temp_col / 3 + 1;

                    board[row_to_write][col_to_write] = '.';
                    if (state[i] > 0)
                    board[row_to_write][col_to_write] = (char)('0' + state[i]);
                }

                println!($"Guessing that {digit1} and {digit2} are arbitrary in {description} (multiple solutions): Pick {final_state[index1]}->({row1 + 1}, {col1 + 1}), {final_state[index2]}->({row2 + 1}, {col2 + 1}).");
            }
        }
        //#endregion

        if (change_made)
        {
            //#region Print the board as it looks after one change was made to it
            println!(string.Join(Environment.NewLine, board.Select(s => new string(s)).ToArray()));
            string
            code =
                string.Join(string.Empty, board.Select(s => new string(s)).ToArray())
                    .Replace("-", string.Empty)
                    .Replace("+", string.Empty)
                    .Replace("|", string.Empty)
                    .Replace(".", "0");

            println!("Code: {0}", code);
            println!();
            //#endregion
        }
    }
}

