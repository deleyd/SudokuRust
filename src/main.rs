use std::collections::{BTreeMap, HashMap};
use std::collections::VecDeque;
use itertools::Itertools;
use std::fs::{File, OpenOptions};
use std::io::{self, Write, BufReader, BufRead};
use std::sync::Mutex;
use std::fs;
use std::io::ErrorKind;
use std::path::Path;

// Declare a global static variable to hold the file.
// Mutex is used for thread-safe access to the file.
// Option is used because the file might not be initialized yet.
static GLOBAL_FILE: Mutex<Option<File>> = Mutex::new(None);


#[derive(Debug, Clone)]
struct Cell {
    discriminator: usize,
    description: String,
    index: usize,
    row: usize,
    column: usize,
}

#[derive(Debug, Clone)]
pub struct CellGroup {
    pub mask: u32,
    pub discriminator: i32,
    pub description: String,
    cells: Vec<Cell>,
}


#[derive(Debug, Clone)]
pub struct CellWithMask<'a> {
    pub mask: u32,
    pub description: String,
    cell_group: (&'a usize, &'a Vec<Cell>),
    cells_with_mask: Vec<Cell>,
    pub cleanable_cells_count: u32,
}

#[derive(Debug, PartialEq)]
enum Commands {
    Expand,
    Move,
    Collapse,
    Complete,
    Fail,
}

fn print_board(board : &[[char; 13]; 13])
{
    for row in board {
        let mut s: String = "".to_owned();
        for &ch in row {
            s.push(ch); // Print each character without a newline
        }
        log(s); // Print a newline after each row
    }
}


fn play(mut rnglcg: PortableLCG) {
    // 1. Prepare empty board
    let line: &str = "+---+---+---+";
    let middle: &str = "|...|...|...|";
    // Declares a 3x4 2D array of characters, initialized with 'X'
    let mut board: [[char; 13]; 13] = [['X'; 13]; 13];

    // Accessing and modifying elements
    let line_chars: Vec<char> = line.chars().collect();
    let middle_chars: Vec<char> = middle.chars().collect();
    board[0] = line_chars.clone().try_into().expect("REASON");
    board[1] = middle_chars.clone().try_into().expect("REASON");
    board[2] = middle_chars.clone().try_into().expect("REASON");
    board[3] = middle_chars.clone().try_into().expect("REASON");
    board[4] = line_chars.clone().try_into().expect("REASON");
    board[5] = middle_chars.clone().try_into().expect("REASON");
    board[6] = middle_chars.clone().try_into().expect("REASON");
    board[7] = middle_chars.clone().try_into().expect("REASON");
    board[8] = line_chars.clone().try_into().expect("REASON");
    board[9] = middle_chars.clone().try_into().expect("REASON");
    board[10] = middle_chars.clone().try_into().expect("REASON");
    board[11] = middle_chars.clone().try_into().expect("REASON");
    board[12] = line_chars.clone().try_into().expect("REASON");
    //print_board(&board);
    //log("EMPTY BOARD!".to_string());

    // 2. Construct board to be solved
    // 3. Top element is current state of the board
    //Stack<int[]> state_stack = new Stack<int[]>();
    let mut state_stack: Vec<[i32; 9 * 9]> = Vec::new(); // Explicitly typed for clarity

    // 4. Top elements are (row, col) of cell which has been modified compared to previous state
    //Stack<int> rowIndexStack = new Stack<int>();
    //Stack<int> colIndexStack = new Stack<int>();
    let mut row_index_stack: Vec<usize> = Vec::new(); // Explicitly typed for clarity
    let mut col_index_stack: Vec<usize> = Vec::new(); // Explicitly typed for clarity

    // 5. Top element indicates candidate digits (those with False) for (row, col)
    //Stack<bool[]> usedDigitsStack = new Stack<bool[]>();
    let mut used_digits_stack: Vec<[bool; 9]> = Vec::new();

    // 6. Top element is the value that was set on (row, col)
    //Stack<int> lastDigitStack = new Stack<int>();
    let mut last_digit_stack: Vec<i32> = Vec::new();

    // 7. Indicates operation to perform next
    // - expand - finds next empty cell and puts new state on stacks
    // - move - finds next candidate number at current pos and applies it to current state
    // - collapse - pops current state from stack as it did not yield a solution
    let mut command = Commands::Expand;

    // we add to the stack each time we add a number to a cell
    while state_stack.len() <= 81  // 8.
    {
        //log("Top of while 8".to_string());
        if command == Commands::Expand
        {
            let current_state: [i32; 81] = if state_stack.len() > 0  // 9.
            {
                state_stack.last().cloned().unwrap()
            } else {
                [0; 81]
            };

            let mut best_row: i32 = -1;
            let mut best_col: i32 = -1;
            let mut best_used_digits: [bool; 9] = [false; 9];
            let mut best_candidates_count: i32 = -1;
            let mut best_random_value: i32 = -1;
            let mut contains_unsolvable_cells: bool = false;
            for index in 0..81  // 10.
            {
                if current_state[index] == 0  // 11.  if cell unused, then let's see what we can do with it
                {
                    let is_digit_used: [bool; 9] = get_row_col_block_used_digits(current_state, index);
                    // for (i = 0..8)

                    // 13.
                    //let candidates_count = is_digit_used.Where(used => !used).Count();
                    let candidates_count: i32 = is_digit_used.iter() // Get an iterator over the elements
                        .filter(|&&value| !value) // Filter for false values
                        .count() as i32; // Count the remaining elements

                    if candidates_count == 0  // 14.
                    {
                        contains_unsolvable_cells = true;
                        break;
                    }

                    let random_value = rnglcg.next();  // 15.

                    if best_candidates_count < 0 ||
                        candidates_count < best_candidates_count ||
                        (candidates_count == best_candidates_count && random_value < best_random_value)
                    {
                        let row: usize = index / 9;
                        let col: usize = index % 9;

                        best_row = row as i32;
                        best_col = col as i32;
                        //println!("13. is_digit_used: {:?}", is_digit_used);
                        best_used_digits = is_digit_used;
                        best_candidates_count = candidates_count;
                        best_random_value = random_value;
                    }
                }
            } // for (i = 0..8)

            if !contains_unsolvable_cells  // 16.
            {
                state_stack.push(current_state);
                row_index_stack.push(best_row as usize);
                col_index_stack.push(best_col as usize);
                //println!("16. best_used_digits: {:?}", best_used_digits);
                used_digits_stack.push(best_used_digits);
                last_digit_stack.push(0); // No digit was tried at this position
                //println!("16. used_digits_stack: {:?}", used_digits_stack);

            }

            // Always try to move after expand
            command = Commands::Move;  // 17.

        } // if (command == Commands::Expand)
        else if command == Commands::Collapse  // 18.
        {
            state_stack.pop();
            row_index_stack.pop();
            col_index_stack.pop();
            used_digits_stack.pop();
            last_digit_stack.pop();

            command = Commands::Move;   // Always try to move after collapse
        }
        else if command == Commands::Move  // 19.
        {
            let rtm = row_index_stack.last().unwrap();
            log(format!("rowIndexStack Count={} rowToMove={}", row_index_stack.len(), rtm));
            if row_index_stack.len() == 20
            {
                let _z = 1;
            }
            if row_index_stack.len() == 0
            {
                //log("row_index_stack.len()= == 0 !!!".to_string());
            }
            let ris_last = row_index_stack.last();
            if ris_last.is_none() {
                //log("ris_last.is_none() !!!".to_string());
            }
            let ris_unwrap = ris_last.unwrap().clone();

            let row_to_move = ris_unwrap;// row_index_stack.last().unwrap();  // panic if empty which it should never be
            let col_to_move = col_index_stack.last().unwrap();
            //println!("19a. last_digit_stack: {:?}", last_digit_stack);
            let digit_to_move: i32 = last_digit_stack.pop().unwrap();

            let row_to_write = row_to_move + row_to_move / 3 + 1;
            let col_to_write = col_to_move + col_to_move / 3 + 1;

            //println!("19b. used_digits_stack.last(): {:?}", used_digits_stack.last());
            let used_digits = used_digits_stack.last_mut().unwrap();
            let current_state = state_stack.last_mut().unwrap();
            let current_state_index = 9 * row_to_move + col_to_move;

            let mut moved_to_digit = digit_to_move + 1;
            //println!("19c. used_digits: {:?} moved_to_digit={:?}", used_digits, moved_to_digit);

            while moved_to_digit <= 9 && used_digits[moved_to_digit as usize - 1]
            {
                moved_to_digit += 1;
            }

            log(format!("digitToMove:{0} movedToDigit:{1} rowToMove:{2} colToMove:{3} rowToWrite:{4} colToWrite:{5} currentStateIndex:{6}", digit_to_move, moved_to_digit, row_to_move, col_to_move, row_to_write, col_to_write, current_state_index));

            if digit_to_move > 0
            {
                used_digits[digit_to_move as usize - 1] = false;
                current_state[current_state_index] = 0;
                board[row_to_write][col_to_write] = '.';
            }

            if moved_to_digit <= 9
            {
                if moved_to_digit == 9
                {
                    let _z = 1;
                }
                log(format!("19d. moved_to_digit: {:?}", moved_to_digit));
                last_digit_stack.push(moved_to_digit);
                used_digits[moved_to_digit as usize - 1] = true;  // DWD This needs to modify the value in *used_digits_stack.last()[moved_to_digit as usize - 1]
                //println!("19e. used_digits: {:?}", used_digits);
                //println!("19e. used_digits_stack: {:?}", used_digits_stack);
                current_state[current_state_index] = moved_to_digit;
                //println!("19f. New Board Value: {} current_state_index={} current_state={:?}",moved_to_digit, current_state_index, current_state);
                board[row_to_write][col_to_write] = char::from_u32((b'0' as i32 + moved_to_digit) as u32).expect("REASON");
                //println!("19g. row={}, col={}, Board: {:?}", row_to_write, col_to_write, board);

                // Next possible digit was found at current position
                // Next step will be to expand the state
                command = Commands::Expand;
            } else {
                // No viable candidate was found at current position - pop it in the next iteration
                last_digit_stack.push(0);
                command = Commands::Collapse;
                log(format!("collapse. last_digit_stack.last():{}", last_digit_stack.last().unwrap().clone()));
            }
        } // if (command == Commands::Move)
    }

    // 20.
    log("".to_string());
    log("Final look of the solved board:".to_string());
    print_board(&board);
    //#endregion

    //#region Generate initial board from the completely solved one
    // Board is solved at this point.
    // Now pick subset of digits as the starting position.
    let remaining_digits : usize = 30;
    let max_removed_per_block = 6;
    let mut removed_per_block: [[i32; 3]; 3] = [[0; 3]; 3];
    //int[] positions = Enumerable.Range(0, 9 * 9).ToArray();
    let mut positions: [usize; 9 * 9] = std::array::from_fn(|i| i);
    let state = state_stack.last_mut().unwrap();

    let final_state = state.clone(); // new int[state.len()]; Array.Copy(state, final_state, final_state.len());

    let mut removed_pos : usize = 0;
    //println!("remaining_digits {:?}",remaining_digits);
    while removed_pos < 9 * 9 - remaining_digits  // 21.
    {
        //println!("positions {:?}",positions);
        let cur_remaining_digits : i32 = (positions.len() - removed_pos) as i32;
        let index_to_pick = removed_pos + rnglcg.next_range(cur_remaining_digits) as usize;

        let row: usize = positions[index_to_pick] / 9;
        let col: usize = positions[index_to_pick] % 9;

        let block_row_to_remove = row / 3;
        let block_col_to_remove = col / 3;

        if removed_per_block[block_row_to_remove][block_col_to_remove] >= max_removed_per_block
        {
            continue;
        }

        removed_per_block[block_row_to_remove][block_col_to_remove] += 1;
        // 21.
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

    // 23
    log("".to_string());
    log("Starting look of the board to solve:".to_string());
    print_board(&board);
    //#endregion

    // 24.
    //#region Prepare lookup structures that will be used in further execution
    log("".to_string());
    let s = "=".repeat(80);
    log(s);
    log("".to_string());

    // 25.
    //Dictionary<int, int> maskToOnesCount = new Dictionary<int, int>();
    let mut mask_to_ones_count: BTreeMap<u32, usize> = BTreeMap::new();
    mask_to_ones_count.insert(0, 0);
    for i in 1..(1 << 9)
    {
        let smaller : u32 = i >> 1;
        let increment : usize = (i & 1) as usize;
        let usize_value = mask_to_ones_count[&smaller] + increment;
        mask_to_ones_count.insert(i, usize_value);
    }
    //println!("mask_to_ones_count: {:?}", mask_to_ones_count);

    // 26.
    //Dictionary < int, int > single_bit_to_index = new
    // single bit to digit might be a better name.
    // 0x00100 = 4 is bit 2? If we start counting from bit 0?
    // converts mask to bit number?
    let mut single_bit_to_index: HashMap<usize, usize> = HashMap::new();
    for i in 0..9
    {
        single_bit_to_index.insert(1 << i, i);
    }
    //println!("single_bit_to_index: {:?}", single_bit_to_index);

    let all_ones : u32 = (1 << 9) - 1;
    //println!("all_ones: {:?}", all_ones);

    //#endregion

    let mut change_made: bool = true;
    while change_made// 27
    {
        change_made = false;

        //#region Calculate candidates for current state of the board
        let mut candidate_masks: [u32; 81] = [0; 81];

        // go through every cell of board, calculcate candicate numbers for each cell that does not already have a number in it.
        // candidate numbers are any digit 1-9 not already in that cell's row, column, and block
        // candidate numbers are stored as a bitmask where bits 0-8 represent digits 1-9
        for i in 0..state.len()  // 28.
        {
            // if this cell doesn't already have a number in it,
            // then calculate all the numbers which can go in this cell
            // candidate_mask is bits 0-8 representing numbers 1-9
            // Array candidate_masks[81] is all candidate numbers for each cell of board
            if state[i] == 0
            {
                let row = i / 9;
                let col = i % 9;
                let block_row = row / 3;
                let block_col = col / 3;

                let mut colliding_numbers : u32 = 0;
                for j in 0..9
                {
                    let row_sibling_index = 9 * row + j;
                    let col_sibling_index = 9 * j + col;
                    let block_sibling_index = 9 * (block_row * 3 + j / 3) + block_col * 3 + j % 3;
// state[81] holds numbers for each cell (or 0 if no number set yet).
                    let row_shift_amount = if state[row_sibling_index] == 0 { 31 } else { state[row_sibling_index] - 1 };
                    let col_shift_amount = if state[col_sibling_index] == 0 { 31 } else { state[col_sibling_index] - 1 };
                    let block_shift_amount = if state[block_sibling_index] == 0 { 31 } else { state[block_sibling_index] - 1 };
// mask has one bit set indicating the digit in state cell. bit 0 = 1, bit 1 = 2,... bit8 = 9. Only bits 0-8 are used.
                    let row_sibling_mask : u32 = 1 << row_shift_amount;
                    let col_sibling_mask : u32 = 1 <<col_shift_amount;
                    let block_sibling_mask : u32 = 1 << block_shift_amount;
// colliding_numbers bits 0-8 indicate what numbers are already present in this cell's row, column, and block.
                    // This is complete list of numbers already used. Stored in one integer.
                    colliding_numbers = colliding_numbers | row_sibling_mask | col_sibling_mask | block_sibling_mask;
                }
// candidate_mask is all numbers available to put in cell[i] of board (state)
                // all the not "already used" numbers
                candidate_masks[i] = all_ones & !colliding_numbers; // mask indicating what numbers are available for this cell.
            }
        }
        //#endregion
        //#region Build a collection (named cellGroups) which maps cell indices into distinct groups (rows/columns/blocks)
        // 29.
        // generate cell_groups which isa BTreeMap<int, Vec<Cell>>
        // int is the Cell discriminator. 0-8 is row this cell is in, 9-17 is column, 18-27 is block
        // for discriminator in range 9-17, subtract 9 to get column
        // for discriminator in range 18-27, subtract 18 to get block
        // Group by rows
        // 30.
        // Group elements by row
        let mut temp_list_row = Vec::<Cell>::new();

        // Create the projected items using a for loop instead of Select
        for index in 0..state.len() {
            let _cell_value = &state[index]; // Note: cellValue is not used in the original code
            let cell = Cell {
                discriminator: index / 9,
                description: format!("row #{}", index / 9 + 1),
                index,
                row: index / 9,
                column: index % 9,
            };
            temp_list_row.push(cell);
        }

        // Group manually using for loops instead of GroupBy
        // Create list of all 81 cells, grouped by row. Discriminator is row, varies from 0 to 8
        let mut group_dictionary_row = HashMap::<usize, Vec<Cell>>::new();
        for cell in temp_list_row {
            let discriminator = cell.discriminator;

            if !group_dictionary_row.contains_key(&discriminator) {
                group_dictionary_row.insert(discriminator, Vec::<Cell>::new());
            }
            group_dictionary_row.get_mut(&discriminator).unwrap().push(cell);
        }

        // Convert dictionary to custom grouping structure
        // THIS MAY NOT BE NECESSARY?
        let mut rows_indices = HashMap::<usize, Vec<Cell>>::new();
        for kvp in group_dictionary_row {
            rows_indices.insert(kvp.0, kvp.1);
        }


        // 31.
        // Group by columns
        // Create list of all 81 cells, grouped by COLUMN. Discriminator is COLUMN, varies from 9 - 17 (subtract 9 to get column)
        let mut temp_list_col = Vec::<Cell>::new();

        // Create the projected items using a for loop instead of Select
        for index in 0..state.len() {
            let _cell_value = &state[index]; // Note: cellValue is not used in the original code
            let cell = Cell {
                discriminator: 9 + index % 9,
                description: format!("column #{}", index % 9 + 1),
                index,
                row: index / 9,
                column: index % 9,
            };
            temp_list_col.push(cell);
        }

        // Group manually using for loops instead of GroupBy
        let mut group_dictionary_col = HashMap::<usize, Vec<Cell>>::new();
        for cell in temp_list_col {
            let discriminator = cell.discriminator;

            if !group_dictionary_col.contains_key(&discriminator) {
                group_dictionary_col.insert(discriminator, Vec::<Cell>::new());
            }
            group_dictionary_col.get_mut(&discriminator).unwrap().push(cell);
        }

        // Convert dictionary to custom grouping structure
        // NOT NECESSARY?
        let mut column_indices = HashMap::<usize, Vec<Cell>>::new();
        for kvp in group_dictionary_col {
            column_indices.insert(kvp.0, kvp.1);
        }


        // Group by blocks
        // Create list of all 81 cells, grouped by BLOCK. Discriminator is BLOCK, varies from 18-26 (subtract 18 to get BLOCK)
        // 32.
        let mut temp_list_block = Vec::<Cell>::new();

        // Create the projected items using a for loop instead of Select
        for index in 0..state.len() {
            let _cell_value = &state[index]; // Note: cellValue is not used in the original code
            let block_row = index / 9;
            let block_column = index % 9;

            let cell = Cell {
                discriminator: 18 + 3 * (block_row / 3) + block_column / 3,
                description: format!("block ({}, {})", block_row / 3 + 1, block_column / 3 + 1),
                index,
                row: index / 9,
                column: index % 9,
            };
            temp_list_block.push(cell);
        }

        // Group BY DISCRIMINATOR
        let mut group_dictionary_block = HashMap::<usize, Vec<Cell>>::new();
        for i in 0..temp_list_block.len() {
            let cell = temp_list_block[i].clone();
            let discriminator = cell.discriminator;

            if !group_dictionary_block.contains_key(&discriminator) {
                group_dictionary_block.insert(discriminator, Vec::<Cell>::new());
            }
            group_dictionary_block.get_mut(&discriminator).unwrap().push(cell);
        }

        // Convert dictionary to custom grouping structure
        // not necessary?
        let mut block_indices = HashMap::<usize, Vec<Cell>>::new();
        for kvp in group_dictionary_block {
            block_indices.insert(kvp.0, kvp.1);
        }


        // Combine all groups
        // 33.
        let mut cell_groups = BTreeMap::<usize, Vec<Cell>>::new();
        cell_groups.extend(rows_indices);
        cell_groups.extend(column_indices);
        cell_groups.extend(block_indices);
        //#endregion

        // cell_groups has 3x 81 cells. 81 for rows, 81 for columns, 81 for blocks
        // 34.
        let mut step_change_made: bool = true;
        while step_change_made  // 35.
        {
            step_change_made = false;

            //#region Pick cells with only one candidate left
            // note mask_to_ones_count. Each bit represents a digit available for this cell.
            // We want to know how many digits to choose from we have. If only one digit, then use that digit.
            // 36.
            let single_candidate_indices: Vec<usize> = candidate_masks
                .iter()
                .enumerate()
                .filter_map(|(index, &mask)| {
                    let candidates_count = mask_to_ones_count.get(&mask).copied().unwrap_or(0);
                    if candidates_count == 1 {
                        Some(index)
                    } else {
                        None
                    }
                })
                .collect();

            // 37.
            // if we have any cells with only one option digit we can put there, then put it there.
            // if we have more than one cell which only one candidate digit, then RANDOMLY select a cell
            // (cells are identified by their index number)
            if single_candidate_indices.len() > 0
            {
                let pick_single_candidate_index : usize = rnglcg.next_range(single_candidate_indices.len() as i32).try_into().unwrap();
                // single_candidate_index identifies the cell we are talking about
                let single_candidate_index = single_candidate_indices[pick_single_candidate_index];
                // candidate_mask is the one candidate digit we can use in this cell
                let candidate_mask = candidate_masks[single_candidate_index];  // candidate_mask has 1 bit set in range 0-8 indicating which digit we can put in this cell
                // candidate is the one digit we can use in this cell 0-8 (add one to get digit)
                let candidate = single_bit_to_index[&(candidate_mask as usize)]; // candidate is 0-8

                let row = single_candidate_index / 9;
                let col = single_candidate_index % 9;

                let row_to_write = row + row / 3 + 1;
                let col_to_write = col + col / 3 + 1;

                state[single_candidate_index] = candidate as i32 + 1;  // Here's the +1 to convert to a digit 1-9
                let c = char::from_u32(b'1' as u32 + candidate as u32).expect("REASON");
                board[row_to_write][col_to_write] = c;  // board parallels state but stores strings for printing
                candidate_masks[single_candidate_index] = 0; // clear candidates for this cell now that we've set cell to a digit
                change_made = true;  // we made a change to the state and board
                // message to user we set a particular cell to the only digit it could be
                let s = format!("({0}, {1}) can only contain {2}.", row + 1, col + 1, candidate + 1);
                log(s);
            }

            //#endregion*

            //#region Try to find a number which can only appear in one place in a row/column/block
            // 38.
            // if there were no cells which could only be set to a single digit
            if !change_made
            {
                let mut group_descriptions: Vec<String> = Vec::new();
                let mut candidate_row_indices: Vec<usize> = Vec::new();
                let mut candidate_col_indices: Vec<usize> = Vec::new();
                let mut candidates: Vec<i32> = Vec::new();
                // 39.
                // candidate_masks is input
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
                        // 40.
                        for index_in_group in 0..9
                        {
                            let row_state_index = 9 * cell_group + index_in_group;
                            let col_state_index = 9 * index_in_group + cell_group;
                            let block_row_index = (cell_group / 3) * 3 + index_in_group / 3;
                            let block_col_index = (cell_group % 3) * 3 + index_in_group % 3;
                            let block_state_index = block_row_index * 9 + block_col_index;
                            // 41.
                            if (candidate_masks[row_state_index] & mask) != 0
                            {
                                row_number_count += 1;
                                index_in_row = index_in_group;
                            }
                            // 42.
                            if (candidate_masks[col_state_index] & mask) != 0
                            {
                                col_number_count += 1;
                                index_in_col = index_in_group;
                            }
                            // 43.
                            if (candidate_masks[block_state_index] & mask) != 0
                            {
                                block_number_count += 1;
                                index_in_block = index_in_group;
                            }
                        }
                        // 44.
                        if row_number_count == 1
                        {
                            group_descriptions.push(format!("Row #{}", cell_group + 1));
                            candidate_row_indices.push(cell_group);
                            candidate_col_indices.push(index_in_row);
                            candidates.push(digit);
                        }
                        // 45.
                        if col_number_count == 1
                        {
                            group_descriptions.push(format!("Column #{}", cell_group + 1));
                            candidate_row_indices.push(index_in_col);
                            candidate_col_indices.push(cell_group);
                            candidates.push(digit);
                        }
                        // 46.
                        if block_number_count == 1
                        {
                            let block_row = cell_group / 3;
                            let block_col = cell_group % 3;

                            group_descriptions.push(format!("Block ({}, {})", block_row + 1, block_col + 1));
                            candidate_row_indices.push(block_row * 3 + index_in_block / 3);
                            candidate_col_indices.push(block_col * 3 + index_in_block % 3);
                            candidates.push(digit);
                        }
                    } // for (cell_group = 0..8)
                } // for (digit = 1..9)

                // 47
                if candidates.len() > 0
                {
                    let index = rnglcg.next_range(candidates.len() as i32) as usize;
                    let description = group_descriptions.get(index).unwrap();
                    let row = candidate_row_indices.get(index).unwrap();
                    let col = candidate_col_indices.get(index).unwrap();
                    let digit = candidates.get(index).unwrap();
                    let row_to_write = row + row / 3 + 1;
                    let col_to_write = col + col / 3 + 1;

                    let state_index = 9 * row + col;
                    state[state_index] = *digit;      // we can try digit in this cell
                    candidate_masks[state_index] = 0;
                    let c = char::from_u32((b'0' as i32 + digit) as u32).expect("REASON");
                    //println!("47. c={}",c);
                    board[row_to_write][col_to_write] = c;

                    change_made = true;

                    let message = format!("{} can contain {} only at ({}, {}).", description, digit, row + 1, col + 1);
                    log(message);
                }
            }
            //#endregion

            //#region Try to find pairs of digits in the same row/column/block and remove them from other colliding cells
            // 48.
            if !change_made
            {
                //let two_digit_masks = candidate_masks.Where(mask => mask_to_ones_count[mask] == 2).Distinct().ToList();
                // look for cells which have only 2 options for digits
                let two_digit_masks1: Vec<u32> = candidate_masks
                    .into_iter()
                    .filter(|&mask| mask_to_ones_count[&mask] == 2)
                    .collect();
                let two_digit_masks: Vec<u32>  = two_digit_masks1.into_iter().unique().collect();
                // note every number here when expressed in biary uses only 2 bits
                log(format!("two_digit_masks={:?}", two_digit_masks));

                // 49.

                let mut groups = Vec::new();

                // Outer loop equivalent to SelectMany over twoDigitMasks
                // for every
                for mask in &two_digit_masks
                {
                    // Inner processing equivalent to the SelectMany lambda
                    for tuple_kvp in &cell_groups
                    {
                        // First Where condition: group.Count(tuple => candidateMasks[tuple.Index] == mask) == 2
                        let mut matching_count = 0;
                        let cell_list = tuple_kvp.1;
                        for cell in cell_list {
                            if candidate_masks[cell.index] == *mask {  // mask represents the two digits (unknown which cell)
                                matching_count += 1;
                            }
                        }
                        // we are hoping to find only 2 cells which can have these two digits
                        if matching_count != 2 {
                            continue;
                        }

                        // only 2 cells confirmed for 2 digits in mask
                        // Second Where condition: group.Any(tuple => candidateMasks[tuple.Index] != mask && (candidateMasks[tuple.Index] & mask) > 0)
                        let mut has_overlapping_non_matching = false;
                        for cell in cell_list {
                            if candidate_masks[cell.index] != *mask && (candidate_masks[cell.index] & mask) > 0 {
                                has_overlapping_non_matching = true;
                                break;
                            }
                        }

                        if !has_overlapping_non_matching {
                            continue;
                        }


                        // Get first description from the group
                        let mut description = String::new();
                        if let Some(first_cell) = cell_list.first() {
                            description = first_cell.description.clone();
                        }

                        // Select equivalent: create the anonymous object equivalent
                        let cell_group = CellGroup {
                            mask: *mask,
                            discriminator: tuple_kvp.0.clone() as i32,
                            description,
                            cells: cell_list.clone(),
                        };

                        groups.push(cell_group);
                    }


                    // 50.
                    if !groups.is_empty()
                    {
                        //log("50. Groups is NOT empty".to_string());
                        for group in groups.iter().sorted_by_key(|cell_group| cell_group.discriminator)
                        {
                            // Translation of the original C# code
                            let mut cells: Vec<Cell> = Vec::new();

                            for cell in &group.cells {
                                if candidate_masks[cell.index] != group.mask &&
                                    (candidate_masks[cell.index] & group.mask) > 0 {
                                    cells.push(cell.clone());
                                }
                            }
                            let mut cells: Vec<_> = group.cells
                                .iter()
                                .filter(|cell| {
                                    candidate_masks[cell.index] != group.mask &&
                                        (candidate_masks[cell.index] & group.mask) > 0
                                })
                                .sorted_by_key(|cell| cell.index)
                                .collect::<Vec<_>>();

                            for cell in &group.cells {
                                if candidate_masks[cell.index] != group.mask &&
                                    (candidate_masks[cell.index] & group.mask) > 0 {
                                    cells.push(cell);
                                }
                            }
                            let cells: Vec<Cell> = group.cells.clone()
                                .iter()
                                .filter(|cell| {
                                    candidate_masks[cell.index] != group.mask &&
                                        (candidate_masks[cell.index] & group.mask) > 0
                                })
                                .cloned() // or .copied() depending on the type of cell and if you need to clone or copy it
                                .collect();

                            let mask_cells: Vec<Cell> = group.cells.clone()
                                .into_iter()
                                .filter(|cell| candidate_masks[cell.index] == group.mask)
                                .map(|x| x)
                                .collect();

                            // 51.
                            if !cells.is_empty()
                            {
                                // "Values {lower} and {upper} in {} are in cells ({mask_cells[0].row+1}, {mask_cells[0].col+1}) and ({mask_cells[1].row+1}, {mask_cells[1].col+1}).",
                                let mut upper = 0;
                                let mut lower = 0;
                                let mut temp = group.mask;  // bits represent digits
// what the hell does this do? Find the upper two bits. upper & lower represent digits
                                let mut value = 1;
                                while temp > 0
                                {
                                    if (temp & 1) > 0  // if low bit is set
                                    {
                                        lower = upper;
                                        upper = value;
                                    }
                                    temp = temp >> 1;  // shift to test next bit in temp
                                    value += 1;  // counts number of bits we've tested
                                }

                                let s = format!(
                                    "Values {} and {} in {} are in cells ({}, {}) and ({}, {}).",
                                    lower,
                                    upper,
                                    group.description,
                                    mask_cells[0].row + 1,
                                    mask_cells[0].column + 1,
                                    mask_cells[1].row + 1,
                                    mask_cells[1].column + 1
                                );
                                log(s);

                                // 52.
                                for cell in &cells
                                {
                                    let mut mask_to_remove = candidate_masks[cell.index] & group.mask;
                                    let mut values_to_remove: Vec<i32> = Vec::new();
                                    let mut cur_value: i32 = 1;
                                    while mask_to_remove > 0
                                    {
                                        if (mask_to_remove & 1) > 0
                                        {
                                            values_to_remove.push(cur_value);
                                        }
                                        mask_to_remove = mask_to_remove >> 1;
                                        cur_value += 1;
                                    }

                                    //string valuesReport = string.Join(", ", values_to_remove.ToArray());
                                    let string_values_to_remove: Vec<String> = values_to_remove
                                        .iter()
                                        .map(|&num| num.to_string())
                                        .collect();
                                    let values_report = string_values_to_remove.join(", ");
                                    let s = format!("{} cannot appear in ({}, {}).", values_report, cell.row + 1, cell.column + 1);

                                    if cell.row + 1 == 7 && cell.column + 1 == 4 //s == "4 cannot appear in (7, 4)"
                                    {
                                        println!("Found cannot appear in (7, 4).")
                                    }
                                    log(s);
                                    candidate_masks[cell.index] &= !group.mask;
                                    step_change_made = true;
                                }
                            }
                        }
                    }
                }
            }
            //#endregion

            //#region Try to find groups of digits of size N which only appear in N cells within row/column/block
            // When a set of N digits only appears in N cells within row/column/block, then no other digit can appear in the same set of cells
            // All other candidates can then be removed from those cells
            // 53.
            if !change_made && !step_change_made
            {
                /*
                IEnumerable < int > masks =
                    mask_to_ones_count
                        .Where(tuple => tuple.Value > 1)
                        .Select(tuple => tuple.Key).ToList(); */
                let masks: Vec<u32> = mask_to_ones_count
                    .iter()
                    .filter(|&(_, count)| *count > 1)
                    .map(|(mask, _)| *mask)
                    .collect();

                let groups_with_n_masks : Vec<CellWithMask> = masks
                    .iter()
                    .flat_map(|mask| {
                        cell_groups
                            .iter()
                            .filter(|cell_group1| {
                                cell_group1.1.iter().all(|cell| {
                                    state[cell.index] == 0
                                        || (mask.clone() & (1 << (state[cell.index] - 1))) == 0
                                })
                            })
                            .map(|cell_group2| {
                                let cells_with_mask: Vec<Cell> = cell_group2.1
                                    .iter()
                                    .filter(|cell| {
                                        state[cell.index] == 0
                                            && (candidate_masks[cell.index] & mask.clone()) != 0
                                    })
                                    //.map(|&x| x)
                                    .cloned()
                                    .collect();

                                let cleanable_cells_count : u32 = cell_group2.1
                                    .iter()
                                    .filter(|cell| {
                                        state[cell.index] == 0
                                            && (candidate_masks[cell.index] & mask.clone()) != 0
                                            && (candidate_masks[cell.index] & !mask.clone()) != 0
                                    })
                                    .count() as u32;

                                CellWithMask {
                                    mask: *mask,
                                    description: cell_group2.1.iter().next().unwrap().description.clone(),
                                    cell_group: cell_group2.clone(),
                                    cells_with_mask,
                                    cleanable_cells_count,
                                }
                            })
                    })
                    .filter(|group| group.cells_with_mask.len() == *mask_to_ones_count.get(&group.mask).unwrap())
                    .collect();

                // 54.
                for group_with_n_masks in groups_with_n_masks
                {
                    let mask = group_with_n_masks.mask;

                    if group_with_n_masks.cell_group.1.iter().any(|cell| {
                        let candidate_mask_for_cell = candidate_masks[cell.index];
                        (candidate_mask_for_cell & mask) != 0 && (candidate_mask_for_cell & !mask) != 0
                    })
                    {
                        let mut message = format!("In {} values ", group_with_n_masks.description);

                        let mut separator = "";
                        let mut temp = mask;
                        let mut cur_value = 1;
                        while temp > 0
                        {
                            if (temp & 1) > 0
                            {
                                let s = format!("{}{}", separator, cur_value);
                                message.push_str(&s);
                                separator = ", ";
                            }
                            temp = temp >> 1;
                            cur_value += 1;
                        }

                        // 55.
                        message.push_str(&" appear only in cells".to_string());
                        for cell in group_with_n_masks.cells_with_mask.clone()
                        {
                            message.push_str(&format!(" ({}, {})", cell.row + 1, cell.column + 1));
                        }

                        // 56.
                        message.push_str(&" and other values cannot appear in those cells.".to_string());

                        log(message);
                    }

                    // 57.
                    for cell in group_with_n_masks.cells_with_mask
                    {
                        let mut mask_to_clear = candidate_masks[cell.index] & !group_with_n_masks.mask;
                        if mask_to_clear == 0
                        {
                            continue;
                        }

                        candidate_masks[cell.index] &= group_with_n_masks.mask;
                        step_change_made = true;

                        let mut value_to_clear = 1;

                        let mut separator: String = "".to_string();
                        let mut message: String = "".to_string();

                        // 58.
                        while mask_to_clear > 0
                        {
                            if mask_to_clear & 1 > 0
                            {
                                message.push_str(&format!("{}{}", separator, value_to_clear));
                                separator = ", ".to_string();
                            }
                            mask_to_clear = mask_to_clear >> 1;
                            value_to_clear += 1;
                        }

                        // 59.
                        message.push_str(&format!(" cannot appear in cell ({}, {}).", cell.row + 1, cell.column + 1));
                        log(message);
                    }
                }
            }
            //#endregion
        } // end while

        //60.
        //#region Final attempt - look if the board has multiple solutions
        if !change_made
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

            // 61.
            for i in 0..candidate_masks.len() - 1
            {
                // 62.
                if mask_to_ones_count[&(candidate_masks[i])] == 2
                {
                    let row = i / 9;
                    let col = i % 9;
                    let block_index = 3 * (row / 3) + col / 3;

                    let mut temp = candidate_masks[i];
                    let mut lower = 0;
                    let mut upper = 0;
                    let mut digit = 1;
                    while temp > 0
                    //for digit in 1..
                    //temp > 0;
                    //digit + +)
                    {
                        if (temp & 1) != 0
                        {
                            lower = upper;
                            upper = digit;
                        }
                        temp = temp >> 1;
                        digit += 1;
                    }

                    // 63.
                    for j in i + 1..candidate_masks.len()
                    {
                        if candidate_masks[j] == candidate_masks[i]
                        {
                            let row1 = j / 9;
                            let col1 = j % 9;
                            let block_index1 = 3 * (row1 / 3) + col1 / 3;

                            if row == row1 || col == col1 || block_index == block_index1
                            {
                                candidate_index1.push_back(i as i32);
                                candidate_index2.push_back(j as i32);
                                candidate_digit1.push_back(lower);
                                candidate_digit2.push_back(upper);
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
            let mut value1: Vec<i32> = Vec::new();
            let mut value2: Vec<i32> = Vec::new();

            // 64.
            while !candidate_index1.is_empty()
            {
                let index1 = candidate_index1.pop_front().unwrap() as usize;
                let index2 = candidate_index2.pop_front().unwrap() as usize;
                let digit1 = candidate_digit1.pop_front().unwrap();
                let digit2 = candidate_digit2.pop_front().unwrap();

                let mut alternate_state = state.clone();

                if final_state[index1] == digit1
                {
                    alternate_state[index1] = digit2;
                    alternate_state[index2] = digit1;
                } else {
                    alternate_state[index1] = digit1;
                    alternate_state[index2] = digit2;
                }

                // 65.
                // What follows below is a complete copy-paste of the solver which appears at the beginning of this method
                // However, the algorithm couldn't be applied directly, and it had to be modified.
                // Implementation below assumes that the board might not have a solution.
                //stateStack = new Stack<int[]>();
                //rowIndexStack = new Stack<int>();
                //colIndexStack = new Stack<int>();
                //usedDigitsStack = new Stack<bool[]>();
                //lastDigitStack = new Stack<int>();
                let mut state_stack: Vec<[i32; 81]> = Vec::new();
                let mut row_index_stack: Vec<usize> = Vec::new();
                let mut col_index_stack: Vec<usize> = Vec::new();
                let mut used_digits_stack: Vec<Vec<bool>> = Vec::new();
                let mut last_digit_stack: Vec<i32> = Vec::new();

                // 66.
                command = Commands::Expand;
                while command != Commands::Complete && command != Commands::Fail
                {
                    if command == Commands::Expand
                    {
                        let current_state;

                        if !state_stack.is_empty()
                        {
                            current_state = state_stack.last().unwrap().clone();
                            //Array.Copy(state_stack.Peek(), current_state, current_state.Length);
                        } else {
                            current_state = alternate_state.clone();
                            //Array.Copy(alternate_state, current_state, current_state.Length);
                        }

                        let mut best_row = 9999;
                        let mut best_col = 9999;
                        let mut best_used_digits: Vec<bool> = Vec::new();
                        let mut best_candidates_count : i32 = -1;
                        let mut best_random_value : i32 = -1;
                        let mut contains_unsolvable_cells: bool = false;

                        // 67.
                        for index in 0..current_state.len()
                        {
                            if current_state[index] == 0
                            {
                                let row = index / 9;
                                let col = index % 9;
                                let block_row = row / 3;
                                let block_col = col / 3;

                                let mut is_digit_used: [bool; 9] = [false; 9];

                                // 68.
                                for i in 0..9
                                {
                                    let row_digit = current_state[9 * i + col];
                                    if row_digit > 0
                                    {
                                        is_digit_used[row_digit as usize - 1] = true;
                                    }

                                    let col_digit = current_state[9 * row + i];
                                    if col_digit > 0
                                    {
                                        is_digit_used[col_digit as usize - 1] = true;
                                    }

                                    let block_digit = current_state[(block_row * 3 + i / 3) * 9 + (block_col * 3 + i % 3)];
                                    if block_digit > 0
                                    {
                                        is_digit_used[block_digit as usize - 1] = true;
                                    }
                                } // for (i = 0..8)

                                // candidates_count = is_digit_used.Where(used => !used).Count();
                                let candidates_count = is_digit_used
                                    .iter()  // 1. Get an iterator over the vector
                                    .filter(|&used| !*used) // 2. Filter elements where 'used' is false
                                    .count(); // 3. Count the remaining elements
                                if candidates_count == 0
                                {
                                    contains_unsolvable_cells = true;
                                    break;
                                }

                                // 70.
                                let random_value = rnglcg.next();
                                //let random_value = rng.Next();

                                if best_candidates_count < 0 ||
                                    candidates_count < best_candidates_count as usize ||
                                    (candidates_count == best_candidates_count as usize && random_value < best_random_value)
                                {
                                    best_row = row;
                                    best_col = col;
                                    best_used_digits = is_digit_used.to_vec();
                                    best_candidates_count = candidates_count as i32;
                                    best_random_value = random_value;
                                }
                            }
                        } // for (index = 0..81)

                        // 71.
                        if !contains_unsolvable_cells
                        {
                            state_stack.push(current_state);
                            row_index_stack.push(best_row);
                            col_index_stack.push(best_col);
                            used_digits_stack.push(best_used_digits);
                            last_digit_stack.push(0); // No digit was tried at this position
                        }

                        // Always try to move after expand
                        command = Commands::Move;
                    } // if (command == Commands::Expand)
                    // 72.
                    else if command == Commands::Collapse
                    {
                        state_stack.pop();
                        row_index_stack.pop();
                        col_index_stack.pop();
                        used_digits_stack.pop();
                        last_digit_stack.pop();

                        if !state_stack.is_empty()
                        {
                            command = Commands::Move; // Always try to move after collapse
                        } else {
                            command = Commands::Fail;
                        }
                    }
                    // 73.
                    else if command == Commands::Move
                    {
                        let row_to_move: usize = row_index_stack.last().unwrap().clone();
                        let col_to_move: usize = col_index_stack.last().unwrap().clone();
                        let digit_to_move = last_digit_stack.pop().unwrap();

                        let row_to_write: usize = row_to_move + row_to_move / 3 + 1;
                        let col_to_write: usize = col_to_move + col_to_move / 3 + 1;

                        let mut used_digits = used_digits_stack.last().unwrap().clone();
                        let current_state = state_stack.last_mut().unwrap();
                        let current_state_index: usize = 9 * row_to_move + col_to_move;

                        let mut moved_to_digit = digit_to_move + 1;
                        while moved_to_digit <= 9 && used_digits[moved_to_digit as usize - 1]
                        {
                            moved_to_digit += 1;
                        }

                        // 74.
                        if digit_to_move > 0
                        {
                            used_digits[digit_to_move as usize - 1] = false;
                            current_state[current_state_index] = 0;
                            board[row_to_write][col_to_write] = '.';
                        }

                        // 75.
                        if moved_to_digit <= 9
                        {
                            last_digit_stack.push(moved_to_digit); // Equivalent of C# Push()
                            used_digits[moved_to_digit as usize - 1] = true; // DWD Problem here. Does not change stack
                            //println!("75. used_digits={:?}", used_digits);
                            current_state[current_state_index] = moved_to_digit; // Array access is similar
                            board[row_to_write][col_to_write] = char::from_u32((b'0' as i32 + moved_to_digit) as u32).expect("REASON"); // Converting integer to char

                            /*if (current_state.Any(digit => digit == 0))
                            command = "expand";
                            else
                            command = "complete";*/
                            command = if current_state.iter().any(|&digit| digit == 0) {
                                Commands::Expand
                            } else {
                                Commands::Complete
                            };
                        } else {
                            // 76.
                            // No viable candidate was found at current position - pop it in the next iteration
                            last_digit_stack.push(0);
                            command = Commands::Collapse;
                        }
                    } // if (command == Commands::Move)

                } // while (command != Commands::Complete && command != Commands::Fail)

                // 77.
                if command == Commands::Complete
                {   // Board was solved successfully even with two digits swapped
                    state_index1.push(index1);
                    state_index2.push(index2);
                    value1.push(digit1);
                    value2.push(digit2);
                }
            } // while (candidate_index1.Any())

            // 78.
            if !state_index1.is_empty()
            {
                let pos = rnglcg.next_range(state_index1.len() as i32) as usize;
                let index1 = state_index1[pos];
                let index2 = state_index2[pos];
                let digit1 = value1[pos];
                let digit2 = value2[pos];
                let row1 = index1 / 9;
                let col1 = index1 % 9;
                let row2 = index2 / 9;
                let col2 = index2 % 9;

                let description: String;

                if index1 / 9 == index2 / 9
                {
                    description = format!("row #{}", index1 / 9 + 1);
                } else if index1 % 9 == index2 % 9
                {
                    description = format!("column #{}", index1 % 9 + 1);
                } else {
                    description = format!("block ({}, {})", row1 / 3 + 1, col1 / 3 + 1);
                }

                state[index1] = final_state[index1];
                state[index2] = final_state[index2];
                candidate_masks[index1] = 0;
                candidate_masks[index2] = 0;
                change_made = true;

                // 79.
                for i in 0..state.len()
                {
                    let temp_row = i / 9;
                    let temp_col = i % 9;
                    let row_to_write = temp_row + temp_row / 3 + 1;
                    let col_to_write = temp_col + temp_col / 3 + 1;

                    board[row_to_write][col_to_write] = '.';
                    if state[i] > 0
                    {
                        let c : char = char::from_u32((b'0' as i32 + state[i]) as u32).expect("REASON");
                        //println!("79. c={:?}", c);
                        board[row_to_write][col_to_write] = c;
                    }
                }

                let s = format!("Guessing that {} and {} are arbitrary in {} (multiple solutions): Pick {}->({}, {}), {}->({}, {}).", digit1, digit2, description, final_state[index1], row1 + 1, col1 + 1, final_state[index2], row2 + 1, col2 + 1);
                log(s);
            }
        }
        //#endregion

        // 80.
        if change_made
        {
            //#region Print the board as it looks after one change was made to it
            print_board(&board);
            /*string code =
                string.Join(string.Empty, board.Select(s => new string(s)).ToArray())
                    .Replace("-", string.Empty)
                    .Replace("+", string.Empty)
                    .Replace("|", string.Empty)
                    .Replace(".", "0");*/
            let code: String = board
                .iter()
                .flat_map(|s| s.iter().copied().collect::<Vec<_>>()) // Flatten the characters from each string in 'board'
                .collect::<String>() // Collect them into a single String
                .replace('-', "")
                .replace('+', "")
                .replace('|', "");
                //.replace('.', "0");
            log(format!("Code: {0}", code));
            log("".to_string());
            let s1 = "..54...2...352.84.462378915349652178271834659658...4325.6.4..81..72635949.4185.6.";
            //             ..54...2...352.84.462378915349652178271834659658...4325.6.4...1..7...5.49.4185.6.
            let s2 = code.as_str();
            if s2 == s1
            {
                let _y = 1;
                println!("FOUND MATCHING CODE");
            }
        }
            //#endregion
    }//while change_made// 27
    log("BOARD SOLVED.".to_string())
}

fn get_row_col_block_used_digits(current_state: [i32; 81], index: usize) -> [bool; 9] {
    let row: usize = index / 9;
    let col: usize = index % 9;
    let block_row: usize = row / 3;
    let block_col: usize = col / 3;

    let mut is_digit_used: [bool; 9] = [false; 9];

    //println!("11. current_state={:?}", current_state);
    // gather all digits used in cell's row, column, and block. output is_digit_used. input: current_state, row, col, block_row, block_col
    for i in 0..9  // 12.
    {
        //println!("current_state: {:?} i={}", current_state, i);

        let row_digit = current_state[9 * i + col];
        if row_digit > 0
        {
            is_digit_used[row_digit as usize - 1] = true;
        }

        let col_digit = current_state[9 * row + i];
        if col_digit > 0
        {
            is_digit_used[col_digit as usize - 1] = true;
        }

        let block_digit = current_state[(block_row * 3 + i / 3) * 9 + (block_col * 3 + i % 3)];
        if block_digit > 0
        {
            is_digit_used[block_digit as usize - 1] = true;
        }
    } // for (i = 0..8)
    is_digit_used
}

#[derive(Clone, Copy)]
pub struct PortableLCG {
    seed: u64,
    a: u64, // Multiplier
    c: u64, // Increment
    m_mask: u64, // Modulus mask (for 48-bit modulus: 2^48 - 1)
}

impl PortableLCG {
    pub fn new(seed: u64) -> Self {
        PortableLCG {
            seed,
            a: 25214903917, // POSIX multiplier
            c: 11,          // POSIX increment
            m_mask: (1u64 << 48) - 1, // 2^48 - 1
        }
    }

    // Generates the next random 32-bit integer
    pub fn next(&mut self) -> i32 {
        // Apply the LCG formula, handling overflow with wrapping_mul and wrapping_add,
        // and using a bitmask for the modulus.
        self.seed = (self.seed.wrapping_mul(self.a).wrapping_add(self.c)) & self.m_mask;
        // Extract the upper 32 bits from the 48-bit state.
        (self.seed >> 17) as i32
    }
    pub fn next_range(&mut self,r:i32) -> i32 {
        return ((self.next() as f64 / 0x7FFFFFFF as f64) * r as f64) as i32;
    }
}


fn write_from_function(content: &str) -> io::Result<()> {
    // Lock the mutex to get access to the global file.
    let mut file_guard = GLOBAL_FILE.lock().unwrap();

    // Check if the file is initialized and write to it.
    if let Some(ref mut file) = *file_guard {
        file.write_all(content.as_bytes())?;
    } else {
        // Handle the case where the file is not yet initialized (e.g., error or not set up).
        eprintln!("Error: File not initialized in GLOBAL_FILE.");
        return Err(io::Error::new(io::ErrorKind::Other, "File not initialized"));
    }
    Ok(())
}

fn log(s : String)
{
    println!("{}", s);
    write_from_function(&*s).expect("TODO: panic message");
    write_from_function("\n").expect("TODO: panic message");
}



fn compare_files_line_by_line(file1_path: &str, file2_path: &str) -> io::Result<()> {
    let file1 = File::open(file1_path)?;
    let file2 = File::open(file2_path)?;

    let reader1 = BufReader::new(file1);
    let reader2 = BufReader::new(file2);

    let mut line1_num = 0;
    let mut line2_num = 0;
    let mut differences_found = false;

    // Create iterators for lines in each file
    let mut lines1 = reader1.lines();
    let mut lines2 = reader2.lines();

    loop {

        let line1_opt = lines1.next();
        let line2_opt = lines2.next();
        line1_num += 1;
        line2_num += 1;

        match (line1_opt, line2_opt) {
            (Some(Ok(line1)), Some(Ok(line2))) => {
                //println!("f2 line: {} trimmed_s={}", line2_num, line2);
                if *line1 != line2 {
                    println!("Difference at file1 line {}, file2 line {}:", line1_num, line2_num);
                    println!("  File 1: {}", line1);
                    println!("  File 2: {}", line2);
                    differences_found = true;
                    break;
                }
            },
            (Some(Ok(line1)), None) => {
                println!("Difference at file1 line {}, file2 line {}: File 1 has extra line: {}", line1_num, line2_num, line1);
                differences_found = true;
                break;
            },
            (None, Some(Ok(line2))) => {
                println!("Difference at file1 line {}, file2 line {}: File 2 has extra line: {}", line1_num, line2_num, line2);
                differences_found = true;
                break;
            },
            (None, None) => {
                // End of both files
                break;
            },
            (Some(Err(e)), _) | (_, Some(Err(e))) => {
                return Err(e); // Handle potential I/O errors during line reading
            },
        }
    }


    if !differences_found {
        println!("Files are identical.");
    } else {
        println!("Differences found between files.");
    }

    Ok(())
}

fn main()
{
    let file_path = "rust_output.txt";

    match fs::remove_file(file_path) {
        Ok(_) => {
            println!("File '{}' removed successfully.", file_path);
        }
        Err(e) => {
            // If the error is Kind::NotFound, the file didn't exist, which is acceptable
            if e.kind() == ErrorKind::NotFound {
                println!("File '{}' does not exist, no action needed.", file_path);
            } else {
                // Handle other potential errors during file deletion
                eprintln!("Error removing file '{}': {}", file_path, e);
            }
        }
    }

    // Open the file in main and store it in the global static variable.
    let file = OpenOptions::new()
        .write(true)
        .create(true)
        .append(true) // Open in append mode to add content without overwriting
        .open(file_path);

    // Lock the mutex and store the file.
    *GLOBAL_FILE.lock().unwrap() = Some(file.expect("REASON"));

    for seed in 1..25
    {
        let my_rng = PortableLCG::new(seed);
        log(format!("RUN {}", seed));
        play(my_rng);
    }

    log("THE END!".to_string());
    // Close file by setting the global file to None (this drops the file handle)
    *GLOBAL_FILE.lock().unwrap() = None;

    let reffile_path1 = r"C:\Users\Charlene\OneDrive\Sudoku\ORIG\SudokuKata\bin\Debug\csharp_output.txt";
    let reffile_path2 = r"C:\Users\David\OneDrive\Sudoku\ORIG\SudokuKata\bin\Debug\csharp_output.txt";
    let reffile_path: &str = if Path::new(reffile_path1).exists()  // 9.
    {
        reffile_path1
    } else {
        reffile_path2
    };

    if let Err(e) = compare_files_line_by_line(file_path, reffile_path) {
        eprintln!("Error comparing files: {}", e);
    }
}
