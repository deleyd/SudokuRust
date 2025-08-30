use std::collections::{BTreeMap, HashMap};
use std::collections::VecDeque;
use itertools::Itertools;
use std::fs::{File, OpenOptions};
use std::io::{self, Write, BufReader, BufRead};
use std::sync::Mutex;
use std::fs;
use std::io::ErrorKind;
use std::ops::{Index, IndexMut};
use std::path::Path;
use std::sync::OnceLock;
use std::fmt;

// Declare a global static variable to hold the file.
// Mutex is used for thread-safe access to the file.
// Option is used because the file might not be initialized yet.
static GLOBAL_FILE: Mutex<Option<File>> = Mutex::new(None);

// Cell should have a value
#[derive(Debug, Clone)]
struct Cell {
    //discriminator: usize,
    description: String,
    index: usize,
    digit: i32,
}
impl Cell {
    pub fn new(i:usize, v:i32) -> Cell {
        Cell {
            description: "".to_string(),
            index: i,
            digit: v,
        }
    }
    fn get_row(&self) -> usize {
        self.index / 9
    }
    fn get_column(&self) -> usize {
        self.index % 9
    }
}
impl fmt::Display for Cell {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(f, "{}", self.digit)
    }
}

#[derive(Debug, Clone)]
struct Board {
    cells: Vec<Cell>,
    candidate_cell: usize,
    used_digits: [bool; 9],
    last_digit: i32,
}
impl Board {
    pub fn new() -> Board {
        let cells: Vec<Cell> = (0..81)
            .map(|i| Cell::new(i, 0))
            .collect();
        Board {
            cells,
            candidate_cell: 9999,
            used_digits:  [false; 9],
            last_digit: -2,
        }
    }
}
impl Index<&CandidateCell> for Board {
    type Output = Cell;

    fn index(&self, candidate_cell: &CandidateCell) -> &Self::Output {
        &self.cells[candidate_cell.index]
    }
}
impl IndexMut<&CandidateCell> for Board {
    fn index_mut(&mut self, candidate_cell: &CandidateCell) -> &mut Self::Output {
        &mut self.cells[candidate_cell.index]
    }
}

impl Index<usize> for Board {
    type Output = Cell;
    fn index(&self, index: usize) -> &Self::Output {
        &self.cells[index]
    }
}
impl IndexMut<usize> for Board {
    fn index_mut(&mut self, index: usize) -> &mut Self::Output {
        &mut self.cells[index]
    }
}

#[derive(Debug, Clone)]
pub struct CellGroup1 {
    pub mask: u32,
    pub discriminator: i32,
    pub description: String,
    cells: Vec<Cell>,
}


#[derive(Debug, Clone)]
pub struct CellGroup2<'a> {
    pub mask: u32,
    pub description: String,
    cell_group: (&'a usize, &'a Vec<Cell>),
    cells_with_mask: Vec<Cell>,
    pub cleanable_cells_count: u32,
}

struct CandidateCell {
    index: usize,
    digit: i32,
    description: String,
}
impl CandidateCell {
    pub fn new(&index: &usize, &digit: &i32, description: &String) -> CandidateCell {
        CandidateCell {
            index,
            digit,
            description: description.clone(),
        }
    }
    fn get_row(&self) -> usize {
        self.index / 9
    }
    fn get_column(&self) -> usize {
        self.index % 9
    }

    fn get_block(&self) -> (usize, usize) {
        let row = self.get_row();
        let col = self.get_column();
        let block_row = row / 3;
        let block_col = col / 3;
        return (block_row, block_col)
    }
}


#[derive(Debug, PartialEq)]
enum Commands {
    Expand,
    Move,
    Collapse,
    Complete,
    Fail,
}

fn print_board(state : &Board)
{
    let line : String = "+---+---+---+".to_string();
    log(&line);
    for j in 0..3 {
        for i in 0..3 {
            let k = (3*j + i) * 9;
            let s = format!("|{}{}{}|{}{}{}|{}{}{}|",
                            state[k + 0], state[k + 1], state[k + 2],
                            state[k + 3], state[k + 4], state[k + 5],
                            state[k + 6], state[k + 7], state[k + 8]);
            let t = s.replace('0', ".");

            log(&t); // Print a newline after each row
        }
        log(&line);
    }
}

fn play(mut rnglcg: PortableLCG) {
    // 2. Construct board to be solved
    // 3. Top element is current state of the board
    //Stack<int[]> state_stack = new Stack<int[]>();

    // 4. Top elements are (row, col) of cell which has been modified compared to previous state
    //let mut cell_candidate_stack: Vec<usize> = Vec::new();

    // 5. Top element indicates candidate digits (those with False) for (row, col)
    // let mut used_digits_stack: Vec<[bool; 9]> = Vec::new();

    // 6. Top element is the value that was set on (row, col)

    // 7. Indicates operation to perform next
    // - expand - finds next empty cell and puts new state on stacks
    // - move - finds next candidate number at current pos and applies it to current state
    // - collapse - pops current state from stack as it did not yield a solution


    // we add to the stack each time we add a number to a cell
    let final_board = construct_final_board(&mut rnglcg);

    // 20.
    log(&"".to_string());
    log(&"Final look of the solved board:".to_string());
    print_board(&final_board);
    //#endregion

    //#region Generate initial board from the completely solved one
    // Board is solved at this point.
    // Now pick subset of digits as the starting position.
    let mut board = generate_initial_board(&mut rnglcg, &final_board);

    // 23
    log(&"".to_string());
    log(&"Starting look of the board to solve:".to_string());
    print_board(&board);
    //#endregion

    // 24.
    //#region Prepare lookup structures that will be used in further execution
    log(&"".to_string());
    let s = "=".repeat(80);
    log(&s);
    log(&"".to_string());

    // 25.
    // 26.
    //#endregion

    let mut change_made: bool = true;
    while change_made// 27
    {
        change_made = false;
        let mut board_candidate_masks: [u32; 81] =  calculate_candidates(&board);
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
        let cell_groups : BTreeMap<usize, Vec< crate::Cell >> = get_indices();

        //#endregion


        // cell_groups has 3x 81 cells. 81 for rows, 81 for columns, 81 for blocks
        // 34.
        let mut step_change_made: bool = true;
        while step_change_made  // 35.
        {
            step_change_made = false;

            //#region Pick cells with only one candidate left

            change_made = change_made || set_cell_with_only_one_candidate(&mut rnglcg, &mut board, &mut board_candidate_masks);
            //#endregion*

            //#region Try to find a number which can only appear in one place in a row/column/block
            // 38.
            // if there were no cells which could only be set to a single digit
            if !change_made
            {
                let candidate_cells = generate_candidate_cells(&mut board_candidate_masks);

                // 47
                if candidate_cells.len() > 0
                {
                    let random_cell_index = rnglcg.next_range(candidate_cells.len() as i32) as usize;
                    let random_candidate_cell = candidate_cells.get(random_cell_index).unwrap();

                    board[random_candidate_cell].digit = random_candidate_cell.digit;       // we can try digit in this cell
                    board_candidate_masks[random_candidate_cell.index] = 0;          // clear for this cell since we just set cell to a number
                    change_made = true;

                    let message = format!("{} can contain {} only at ({}, {}).",
                                          random_candidate_cell.description,
                                          random_candidate_cell.digit,
                                          random_candidate_cell.get_row() + 1,
                                          random_candidate_cell.get_column() + 1);
                    log(&message);
                }
            }
            //#endregion

            //#region Try to find pairs of digits in the same row/column/block and remove them from other colliding cells
            // 48.
            if !change_made
            {
                //let two_digit_masks = candidate_masks.Where(mask => mask_to_ones_count[mask] == 2).Distinct().ToList();
                // look for cells which have only 2 options for digits
                let two_digit_masks: Vec<u32> = board_candidate_masks
                    .into_iter()
                    .filter(|&mask| mask_to_ones_count()[&mask] == 2)
                    .unique()
                    .collect();

                // note every number here when expressed in biary uses only 2 bits, indicating there are two candidates for which? cells
                log(&format!("two_digit_masks={:?}", two_digit_masks));

                // 49.

                let mut groups = Vec::new();

                // Outer loop equivalent to SelectMany over twoDigitMasks
                // for every
                for mask in two_digit_masks
                {
                    // Inner processing equivalent to the SelectMany lambda
                    for tuple_kvp in &cell_groups
                    {
                        // First Where condition: group.Count(tuple => candidateMasks[tuple.Index] == mask) == 2
                        let cell_list = tuple_kvp.1;
                        let matching_count = cell_list.iter()
                            .filter(|cell| board_candidate_masks[cell.index] == mask)
                            .count();

                        // we are hoping to find only 2 cells which can have these two digits
                        if matching_count != 2 {
                            continue;
                        }

                        // only 2 cells confirmed for 2 digits in mask
                        // Second Where condition: group.Any(tuple => candidateMasks[tuple.Index] != mask && (candidateMasks[tuple.Index] & mask) > 0)
                        let has_overlapping_non_matching = cell_list.iter().any(|cell| {
                            board_candidate_masks[cell.index] != mask && (board_candidate_masks[cell.index] & mask) > 0
                        });

                        if !has_overlapping_non_matching {
                            continue;
                        }


                        // Get first description from the group
                        let description = cell_list.first().map(|cell| cell.description.clone()).unwrap_or_default();

                        let cell_group = CellGroup1 {
                            mask,
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
                            let cells: Vec<_> = group.cells.iter()
                                .filter(|cell| board_candidate_masks[cell.index] != group.mask && // not equal but overlaps group.mask
                                    (board_candidate_masks[cell.index] & group.mask) > 0)
                                .sorted_by_key(|cell| cell.index)
                                .collect::<Vec<_>>();

                            let mask_cells: Vec<&Cell> = group.cells.iter()
                                .filter(|cell| board_candidate_masks[cell.index] == group.mask) // equal to group.mask
                                .map(|x| x)
                                .collect();

                            // 51.
                            if !cells.is_empty()
                            {
                                // "Values {lower} and {upper} in {} are in cells ({mask_cells[0].row+1}, {mask_cells[0].col+1}) and ({mask_cells[1].row+1}, {mask_cells[1].col+1}).",
                                // Find the upper two bits. upper & lower represent digits
                                let (lower, upper) = top_two_digits(group.mask); // bits represent digits

                                let s = format!(
                                    "Values {} and {} in {} are in cells ({}, {}) and ({}, {}).",
                                    lower,
                                    upper,
                                    group.description,
                                    mask_cells[0].get_row() + 1,
                                    mask_cells[0].get_column() + 1,
                                    mask_cells[1].get_row() + 1,
                                    mask_cells[1].get_column() + 1
                                );
                                log(&s);

                                // 52.
                                for cell in &cells
                                {
                                    let mask_to_remove = board_candidate_masks[cell.index] & group.mask;  // intersection
                                    let values_to_remove = mask_to_vec_digits(mask_to_remove);

                                    //string valuesReport = string.Join(", ", values_to_remove.ToArray());
                                    let string_values_to_remove: Vec<String> = values_to_remove
                                        .iter()
                                        .map(|&num| num.to_string())
                                        .collect();
                                    let values_report = string_values_to_remove.join(", ");
                                    let s = format!("{} cannot appear in ({}, {}).", values_report, cell.get_row() + 1, cell.get_column() + 1);
                                    log(&s);
                                    board_candidate_masks[cell.index] &= !group.mask;
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
                let digit_masks: Vec<u32> = mask_to_ones_count().iter()
                    .filter(|&(_, count)| *count > 1)
                    .map(|(mask, _)| *mask)
                    .collect();
                let groups_with_n_masks : Vec<CellGroup2> = digit_masks
                    .iter()
                    .flat_map(|mask| {
                        cell_groups
                            .iter()
                            .filter(|cell_group1| {
                                cell_group1.1.iter().all(|cell| {
                                    board[cell.index].digit == 0          // cell not being used
                                        || (mask.clone() & convert_digit_to_mask(board[cell.index].digit)) == 0  // cell not using value
                                })
                            })
                            .map(|cell_group2| {
                                let cells_with_mask: Vec<Cell> = cell_group2.1
                                    .iter()
                                    .filter(|cell| {
                                        board[cell.index].digit == 0  // cell unused
                                            && (board_candidate_masks[cell.index] & mask.clone()) != 0  // candidate_mask overlaps mask
                                    })
                                    //.map(|&x| x)
                                    .cloned()
                                    .collect();

                                let cleanable_cells_count : u32 = cell_group2.1
                                    .iter()
                                    .filter(|cell| {
                                        board[cell.index].digit == 0
                                            && (board_candidate_masks[cell.index] & mask.clone()) != 0   // overlaps
                                            && (board_candidate_masks[cell.index] & !mask.clone()) != 0  // but is not equal to
                                    })
                                    .count() as u32;

                                CellGroup2 {
                                    mask: *mask,
                                    description: cell_group2.1.iter().next().unwrap().description.clone(),
                                    cell_group: cell_group2.clone(),
                                    cells_with_mask,
                                    cleanable_cells_count,
                                }
                            }) // .map
                    })// .flat_map
                    .filter(|group| group.cells_with_mask.len() == *mask_to_ones_count().get(&group.mask).unwrap())
                    .collect();

                // 54.
                for group_with_n_masks in groups_with_n_masks
                {
                    let mask = group_with_n_masks.mask;

                    if group_with_n_masks.cell_group.1.iter().any(|cell| {
                        let candidate_mask_for_cell = board_candidate_masks[cell.index];
                        (candidate_mask_for_cell & mask) != 0 && (candidate_mask_for_cell & !mask) != 0  // there is some overlap
                    })
                    {
                        log_group_with_n_masks_message(&group_with_n_masks, mask);
                    }

                    // 57.
                    for cell in group_with_n_masks.cells_with_mask
                    {
                        let mask_to_clear = board_candidate_masks[cell.index] & !group_with_n_masks.mask;  // mask_to_clear is the intersection of
                        if mask_to_clear == 0
                        {
                            continue;
                        }

                        board_candidate_masks[cell.index] &= group_with_n_masks.mask;  // Add more candidate digits to this cell
                        step_change_made = true;

                        generate_and_log_mask_to_clear_message(mask_to_clear, cell);

                        // 59.

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
            let mut candidate_cells1: VecDeque<CandidateCell> = VecDeque::new();
            let mut candidate_cells2: VecDeque<CandidateCell> = VecDeque::new();

            // 61.
            // index i goes from 0 to 80, index j goes from i+1 to 81. Gives us two cells, i & j, to compare candidate digits
            for i in 0..80  // stop at 80 because j looks at i+1 cell
            {
                // 62.
                if mask_to_ones_count()[&(board_candidate_masks[i])] == 2   // if this cell candidate i has exactly 2 digits
                {
                    let (lower_digit, upper_digit) = top_two_digits(board_candidate_masks[i]); // we already determined that this candidate has exactly 2 digits

                    // 63.
                    for j in i + 1..81
                    {
                        if board_candidate_masks[j] == board_candidate_masks[i]    // if candidate digits for cells[i] & cells[j] are identical set,
                        {
                            if row_or_column_or_block_overlap(i,j)
                            {
                                candidate_cells1.push_back(CandidateCell::new(&i, &lower_digit, &"".to_string()));
                                candidate_cells2.push_back(CandidateCell::new(&j, &upper_digit, &"".to_string()));
                            }
                        }
                    }
                }
            }

            // At this point we have the lists with pairs of cells that might pick one of two digits each
            // Now we have to check whether that is really true - does the board have two solutions?
            // possibly replace with cellList1,2. holding cells and values set for cell.
            let mut candidate_cells_stack1: Vec<CandidateCell> = Vec::new();
            let mut candidate_cells_stack2: Vec<CandidateCell> = Vec::new();

            // 64.
            while !candidate_cells1.is_empty()
            {
                let candidate_cell1 = candidate_cells1.pop_front().unwrap();
                let candidate_cell2 = candidate_cells2.pop_front().unwrap();

                let mut alternate_board : Board = board.clone();

                // assign digit1, digit2, in the order opposite of final_board
                if final_board[candidate_cell1.index].digit == candidate_cell1.digit
                {
                    alternate_board[&candidate_cell1].digit = candidate_cell2.digit;
                    alternate_board[&candidate_cell2].digit = candidate_cell1.digit;
                } else {
                    alternate_board[candidate_cell1.index].digit = candidate_cell1.digit;
                    alternate_board[candidate_cell2.index].digit = candidate_cell2.digit;
                }

                // 65.
                // What follows below is a complete copy-paste of the solver which appears at the beginning of this method
                // However, the algorithm couldn't be applied directly, and it had to be modified.
                // Implementation below assumes that the board might not have a solution.
                let mut board_stack: Vec<Board> = Vec::new();
                //let mut cell_candidate_stack: Vec<usize> = Vec::new();
                //let mut used_digits_stack: Vec<[bool; 9]> = Vec::new();
                //let mut last_digit_stack: Vec<i32> = Vec::new();

                // 66.
                let mut command = Commands::Expand;
                while command != Commands::Complete && command != Commands::Fail
                {
                    if command == Commands::Expand
                    {
                        let mut current_board : Board = if !board_stack.is_empty() {
                            board_stack.last_mut().unwrap().clone()
                        } else {
                            alternate_board.clone()
                        };

                        let mut best_index: usize = 9999;                  // best will be cell with lowest number of candidate digits
                        let mut best_used_digits: [bool; 9] = [false; 9];  // corresponding candidate digits for best cell
                        let mut best_candidates_count : i32 = -1;          // number of candidate digits for best cell
                        let mut best_random_value : i32 = -1;
                        let mut contains_unsolvable_cells: bool = false;

                        // 67.
                        for index in 0..81
                        {
                            if current_board[index].digit == 0
                            {
                                // 68.
                                let digit_used_array = gather_digits(&current_board, index);

                                let candidates_count : i32 = digit_used_array
                                    .iter()  // 1. Get an iterator over the vector
                                    .filter(|&used| !*used) // 2. Filter elements where 'used' is false
                                    .count() as i32; // 3. Count the remaining elements
                                if candidates_count == 0
                                {
                                    contains_unsolvable_cells = true;
                                    break;
                                }

                                // 70.
                                let random_value = rnglcg.next();
                                //let random_value = rng.Next();

                                if best_candidates_count < 0 ||
                                    candidates_count < best_candidates_count ||
                                    (candidates_count == best_candidates_count && random_value < best_random_value)
                                {
                                    best_index = index;                               // corresponding cell with lowest number of candidate digits
                                    best_used_digits = digit_used_array;                 // the candidate digits for this cell
                                    best_candidates_count = candidates_count as i32;  // best is lowest number of candidate digits for a cell
                                    best_random_value = random_value;
                                }
                            }
                        } // for (index = 0..81)

                        // 71.
                        if !contains_unsolvable_cells
                        {
                            current_board.candidate_cell = best_index;
                            current_board.used_digits = best_used_digits;
                            current_board.last_digit = 0;
                            board_stack.push(current_board);

                            //cell_candidate_stack.push(best_index);      // CellCandidate is index of cell on Board
                            //used_digits_stack.push(best_used_digits);   // corresponding digits already used for cell's row,col,block
                            //last_digit_stack.push(0); // No digit was tried at this position. Last digit tried for this cell
                        }

                        // Always try to move after expand
                        command = Commands::Move;
                    } // if (command == Commands::Expand)
                    // 72.
                    else if command == Commands::Collapse
                    {
                        board_stack.pop();

                        command = if !board_stack.is_empty() {
                            Commands::Move // Always try to move after collapse
                        } else {
                            Commands::Fail
                        }
                    }
                    // 73.
                    else if command == Commands::Move
                    {
                        let cell_to_move: usize = board_stack.last_mut().unwrap().candidate_cell;  // cell to move is identified by an index
                        let digit_to_move = board_stack.last_mut().unwrap().last_digit;  // pop here, push below
                        //let mut used_digits : [bool; 9] = used_digits_stack.last().unwrap().clone();

                        //let current_board = board_stack.last_mut().unwrap();
                        let current_cell_index: usize = cell_to_move;

                        let mut moved_to_digit = digit_to_move + 1;
                        // Find next digit not used
                        while moved_to_digit <= 9 && board_stack.last_mut().unwrap().used_digits[moved_to_digit as usize - 1]
                        {
                            moved_to_digit += 1;
                        }

                        // 74.
                        if digit_to_move > 0
                        {
                            //used_digits[digit_to_move as usize - 1] = false;
                            board_stack.last_mut().unwrap().used_digits[digit_to_move as usize - 1] = false;
                            board_stack.last_mut().unwrap()[current_cell_index].digit = 0;  // set cell to unused
                        }

                        // 75.
                        if moved_to_digit <= 9
                        {
                            board_stack.last_mut().unwrap().last_digit = moved_to_digit;
                            board_stack.last_mut().unwrap().used_digits[moved_to_digit as usize - 1] = true;

                            //last_digit_stack.pop();
                            //last_digit_stack.push(moved_to_digit); // Equivalent of C# Push()
                            //used_digits[moved_to_digit as usize - 1] = true; // DWD Problem here. Does not change stack
                            board_stack.last_mut().unwrap()[current_cell_index].digit = moved_to_digit; // Array access is similar

                            command = if board_stack.last_mut().unwrap().cells.iter().any(|cell| cell.digit == 0) {
                                Commands::Expand
                            } else {
                                Commands::Complete
                            };
                        } else {
                            // 76.
                            // No viable candidate was found at current position - pop it in the next iteration
                            board_stack.last_mut().unwrap().last_digit = 0;
                            command = Commands::Collapse;
                        }
                    } // if (command == Commands::Move)
                } // while (command != Commands::Complete && command != Commands::Fail)

                // 77.
                if command == Commands::Complete
                {   // Board was solved successfully even with two digits swapped
                    // sync state_index with value
                    // state_index : an array of indexes (indexes are cells, cells have values). value : array of values corresponding to array of indexes.
                    push_candidate_cell(&mut candidate_cells_stack1, candidate_cell1);
                    push_candidate_cell(&mut candidate_cells_stack2, candidate_cell2);
                }
            } // while (candidate_index1.Any())

            // 78.
            if !candidate_cells_stack1.is_empty()
            {
                let random_pos = rnglcg.next_range(candidate_cells_stack1.len() as i32) as usize;
                let candidate_cell1 = &candidate_cells_stack1[random_pos];
                let candidate_cell2 = &candidate_cells_stack2[random_pos];

                board[candidate_cell1].digit = final_board[candidate_cell1].digit;
                board[candidate_cell2].digit = final_board[candidate_cell2].digit;
                board_candidate_masks[candidate_cell1.index] = 0;
                board_candidate_masks[candidate_cell2.index] = 0;
                change_made = true;

                // 79.
                // print
                let row1 = candidate_cell1.get_row();
                let col1 = candidate_cell1.get_column();
                let row2 = candidate_cell2.get_row();
                let col2 = candidate_cell2.get_column();
                let description: String;

                if row1 == row2
                {
                    description = format!("row #{}", row1 + 1);
                } else if col1 == col2
                {
                    description = format!("column #{}", col1 + 1);
                } else {
                    let (block_row, block_col) = candidate_cell1.get_block();
                    description = format!("block ({}, {})", block_row+1, block_col+1);
                }

                let s = format!("Guessing that {} and {} are arbitrary in {} (multiple solutions): Pick {}->({}, {}), {}->({}, {}).",
                                candidate_cell1.digit,
                                candidate_cell2.digit,
                                description,
                                final_board[candidate_cell1.index],
                                row1 + 1,
                                col1 + 1,
                                final_board[candidate_cell2.index],
                                row2 + 1,
                                col2 + 1);
                log(&s);
            }
        }
        //#endregion

        // 80.
        if change_made  // print board and Code if we made a change
        {
            //#region Print the board as it looks after one change was made to it
            // convert this to use state instead of board
            print_board(&board);
            let code: String = board.cells.iter()
                .map(|cell| if cell.digit == 0 { ".".to_string() } else { cell.digit.to_string() })// convert all 0 to .
                .collect();
            log(&format!("Code: {0}", code));
            log(&"".to_string());
        }
            //#endregion
    }//while change_made// 27
    log(&"BOARD SOLVED.".to_string())
}

fn push_candidate_cell(board_stack2: &mut Vec<CandidateCell>, candidate_cell2: CandidateCell) {
    board_stack2.push(candidate_cell2);
}

fn generate_and_log_mask_to_clear_message(mask_to_clear: u32, cell: Cell) {
    let mut mask_to_clear = mask_to_clear;
    let mut value_to_clear = 1;
    let mut separator: String = "".to_string();
    let mut message: String = "".to_string();

    // 58.
    // convert mask to digits and append to message for each digit
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
    message.push_str(&format!(" cannot appear in cell ({}, {}).", cell.get_row() + 1, cell.get_column() + 1));
    log(&message);
}

fn index_to_row(i: usize) -> usize {
    i / 9
}
fn index_to_col(i: usize) -> usize {
    i % 9
}
fn index_to_block_index(i: usize) -> usize {
    3 * (index_to_row(i) / 3) + index_to_col(i) / 3
}
fn index_to_block_row(i: usize) -> usize {
    index_to_row(i) / 3
}
fn index_to_block_col(i: usize) -> usize {
    index_to_col(i) / 3
}

fn index_to_block(i: usize) -> usize {
    let row = index_to_row(i);
    let col = index_to_col(i);
    return 3 * (row / 3) + col / 3
}

fn row_or_column_or_block_overlap(i:usize, j:usize) -> bool {
    let row_i = index_to_row(i);
    let col_i = index_to_col(i);
    let block_i = index_to_block(i);
    let row_j = index_to_row(j);
    let col_j = index_to_col(j);
    let block_j = index_to_block(j);
    return row_i == row_j || col_i == col_j || block_i == block_j
}

fn generate_candidate_cells(board_candidate_masks: &mut [u32; 81]) -> Vec<CandidateCell> {
    let mut candidate_cells: Vec<CandidateCell> = Vec::new();
    // 39.
    // candidate_masks is input
    // test each digit
    for digit in 1..=9
    {
        let digit_mask = convert_digit_to_mask(digit);  // mask representing digit. Convert digit to single bit mask.
        // test every cell in the board
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
                // 41. Check row "cell_group". cell_group covers 0..9 in this case cell_group = row number 0-8
                let row_state_index = 9 * cell_group + index_in_group;
                if (board_candidate_masks[row_state_index] & digit_mask) != 0  // this cell has digit as a candidate
                {
                    row_number_count += 1;  // We check the row and find cells which can be set to digit. Count the number of cells in this row which can be set to digit.
                    index_in_row = index_in_group;
                }
                // 42. Check column "cell_group". cell_group 0..9 In this case cell_group = column 0-8
                let col_state_index = 9 * index_in_group + cell_group;
                if (board_candidate_masks[col_state_index] & digit_mask) != 0  // this cell has digit as a candidate
                {
                    col_number_count += 1;  // We check the column and find cells which can be set to digit. Count the number of cells in this column which can be set to digit.
                    index_in_col = index_in_group;
                }
                // 43. Check block "cell_group". cell_group 0..9 In this case cell_group = block number 0-8
                let block_row_index = (cell_group / 3) * 3 + index_in_group / 3;
                let block_col_index = (cell_group % 3) * 3 + index_in_group % 3;
                let block_state_index = block_row_index * 9 + block_col_index;
                if (board_candidate_masks[block_state_index] & digit_mask) != 0  // this cell has digit as a candidate
                {
                    block_number_count += 1;  // We check the block and find cells which can be set to digit. Count the number of cells in this block which can be set to digit.
                    index_in_block = index_in_group;
                }
            }
            // 44.
            if row_number_count == 1  // If there is only one cell in this row which can be set to digit, push cell on candidate stack.
            {
                candidate_cells.push(CandidateCell::new(&(cell_group * 9 + index_in_row), &digit, &format!("Row #{}", cell_group + 1)))
            }
            // 45.
            if col_number_count == 1  // If there is only one cell in this column which can be set to digit, push cell on candidate stack.
            {
                candidate_cells.push(CandidateCell::new(&(index_in_col * 9 + cell_group), &digit, &format!("Column #{}", cell_group + 1)))
            }
            // 46.
            if block_number_count == 1  // If there is only one cell in this block which can be set to digit, push cell on candidate stack.
            {
                let block_row = cell_group / 3;
                let block_col = cell_group % 3;
                candidate_cells.push(CandidateCell::new(&((block_row * 3 + index_in_block / 3) * 9 + (block_col * 3 + index_in_block % 3)), &digit, &format!("Block ({}, {})", block_row + 1, block_col + 1)));
            }
        } // for (cell_group = 0..8)
    } // for (digit = 1..9)
    candidate_cells
}

fn generate_initial_board(rnglcg: &mut PortableLCG, final_board: &Board) -> Board {
    let remaining_digits : usize = 30;
    let max_removed_per_block = 6;
    let mut removed_per_block: [[i32; 3]; 3] = [[0; 3]; 3];
    let mut positions: [usize; 9 * 9] = std::array::from_fn(|i| i);
    let mut removed_pos: usize = 0;
    let mut board = final_board.clone(); // new int[state.len()]; Array.Copy(state, final_state, final_state.len());

    while removed_pos < 9 * 9 - remaining_digits  // 21.
    {
        let cur_remaining_digits: i32 = (positions.len() - removed_pos) as i32;
        let index_to_pick = removed_pos + rnglcg.next_range(cur_remaining_digits) as usize;

        let picked_index = positions[index_to_pick];
        let row: usize = index_to_row(picked_index);
        let col: usize = index_to_col(picked_index);

        let block_row_to_remove = index_to_block_row(picked_index);
        let block_col_to_remove = index_to_block_col(picked_index);

        if removed_per_block[block_row_to_remove][block_col_to_remove] >= max_removed_per_block
        {
            continue;
        }

        removed_per_block[block_row_to_remove][block_col_to_remove] += 1;
        // 21.
        let cell_index: usize = 9 * row + col;
        board[cell_index].digit = 0;

        // swap [removed_pos] with [index_to_pick]
        let temp = positions[removed_pos];
        positions[removed_pos] = positions[index_to_pick];
        positions[index_to_pick] = temp;
        removed_pos += 1;
    }
    board
}

fn construct_final_board(mut rnglcg: &mut PortableLCG) -> Board {
    let mut command = Commands::Expand;
    let mut board_stack: Vec<Board> = Vec::new();

    while board_stack.len() <= 81  // 8.
    {
        match command {
            Commands::Expand => {
                handle_expand(&mut rnglcg, &mut board_stack);
                command = Commands::Move;  // 17. // Always try to move after expand
            }
            Commands::Collapse => {
                board_stack.pop();
                command = Commands::Move;   // Always try to move after collapse
            }
            Commands::Move => {
                command = handle_move(&mut board_stack);
            }
            _ => {
                // should never get here
                log(&"Fatal Error. command did not match anything.".to_string());
            }
        }
    }
    board_stack.last().unwrap().clone()
}

fn handle_move(board_stack: &mut Vec<Board>) -> Commands {
    let stack_len = board_stack.len();
    log(&format!("rowIndexStack Count={} rowToMove={}", stack_len, board_stack.last().unwrap().candidate_cell / 9));
    let cell_to_move: usize = board_stack.last_mut().unwrap().candidate_cell;
    let digit_to_move: i32 = board_stack.last_mut().unwrap().last_digit;
    let mut moved_to_digit = digit_to_move + 1;

    while moved_to_digit <= 9 && board_stack.last_mut().unwrap().used_digits[moved_to_digit as usize - 1]
    {
        moved_to_digit += 1;
    }

    let row_to_move = index_to_row(cell_to_move);  // row_index_stack.last().unwrap();  // panic if empty which it should never be
    let col_to_move = index_to_col(cell_to_move);  // col_index_stack.last().unwrap();
    let row_to_write: usize = (row_to_move + row_to_move / 3 + 1) as usize;
    let col_to_write: usize = (col_to_move + col_to_move / 3 + 1) as usize;
    log(&format!("digitToMove:{0} movedToDigit:{1} rowToMove:{2} colToMove:{3} rowToWrite:{4} colToWrite:{5} currentStateIndex:{6}", digit_to_move, moved_to_digit, row_to_move, col_to_move, row_to_write, col_to_write, cell_to_move));

    if digit_to_move > 0
    {
        //used_digits[digit_to_move as usize - 1] = false;
        board_stack.last_mut().unwrap().used_digits[digit_to_move as usize - 1] = false;
        board_stack.last_mut().unwrap()[cell_to_move].digit = 0; // does this change last element of state_stack?
    }

    if moved_to_digit <= 9
    {
        log(&format!("19d. moved_to_digit: {:?}", moved_to_digit));
        board_stack.last_mut().unwrap().last_digit = moved_to_digit;
        board_stack.last_mut().unwrap().used_digits[moved_to_digit as usize - 1] = true;
        board_stack.last_mut().unwrap()[cell_to_move].digit = moved_to_digit;

        // Next possible digit was found at current position
        // Next step will be to expand the state
        // Next step will be to expand the state/ 9
        return Commands::Expand;
    } else {
        // No viable candidate was found at current position - pop it in the next iteration
        board_stack.last_mut().unwrap().last_digit = 0;
        log(&format!("collapse. last_digit_stack.last():{}", board_stack.last_mut().unwrap().last_digit));
        return Commands::Collapse;
    }
}

fn handle_expand(rnglcg: &mut PortableLCG, board_stack: &mut Vec<Board>) {
    let mut current_board: Board = if !board_stack.is_empty()   // 9.
    {
        board_stack.last_mut().unwrap().clone()
    } else {
        Board::new()
    };

    // input: current_state. output: contains_unsolvable_cells, best_row, best_col,
    let mut best_index: usize = 9999;
    let mut best_used_digits: [bool; 9] = [false; 9];
    let mut best_candidates_count: i32 = -1;
    let mut best_random_value: i32 = -1;
    let mut contains_unsolvable_cells: bool = false;
    // loop through all cells looking for empty ones
    for index in 0..81  // 10.
    {
        if current_board[index].digit != 0 {
            continue;
        }
        // 11.  otherwise cell unused. Let's see what we can do with it

        let digits_used_array: [bool; 9] = get_row_col_block_used_digits(&current_board, index); // returns an array of 9 true/false values

        // 13.
        let candidates_count: i32 = digits_used_array
            .iter() // Get an iterator over the elements
            .filter(|&&value| !value) // count the 'false' values (filter out 'true' values)
            .count() as i32; // Count the remaining elements

        if candidates_count == 0  // 14.  if there are no candidates, then this cell has no options and Sudoku is unsolvable
        {
            contains_unsolvable_cells = true;
            break;
        }

        let random_value = rnglcg.next();  // 15. random value if we need it

        // if we have no best candidates, or best candidates outnumber candidates, or
        // then update best everything
        if best_candidates_count < 0 ||                  // if we're just starting
            candidates_count < best_candidates_count ||  // looking for the cell with the LEAST number of candidates
            (candidates_count == best_candidates_count && random_value < best_random_value) // if two cells both have the same number of candidates, randomly select one (this "random" looks not random)
        {
            best_index = index; // this cell becomes the best cell (saved as row,col. we could save index instead?)
            best_used_digits = digits_used_array;
            best_candidates_count = candidates_count;  // candidates_count is a function of is_digit_used array
            best_random_value = random_value;
        }
    }

    if !contains_unsolvable_cells  // 16.
    {
        current_board.candidate_cell = best_index;
        current_board.used_digits = best_used_digits;
        current_board.last_digit = 0;
        board_stack.push(current_board);          // current state came from state_stack?
    }
}

fn log_group_with_n_masks_message(group_with_n_masks: &CellGroup2, mask: u32) {
    let mut message = format!("In {} values ", group_with_n_masks.description);
    let mut separator = "";
    let mut temp = mask;
    let mut cur_value = 1;
    // convert mask to message
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
        message.push_str(&format!(" ({}, {})", cell.get_row() + 1, cell.get_column() + 1));
    }

    // 56.
    message.push_str(&" and other values cannot appear in those cells.".to_string());

    log(&message);
}

fn set_cell_with_only_one_candidate(mut rnglcg: &mut PortableLCG, board: &mut Board, mut board_candidate_masks: &mut [u32; 81]) -> bool {
    // note mask_to_ones_count. Each bit represents a digit available for this cell.
    // We want to know how many digits to choose from we have. If only one digit, then use that digit.
    // 36.
    let single_candidate_indices = get_single_candidate_indices(&mut board_candidate_masks);

    // 37.
    // if we have any cells with only one option digit we can put there, then put it there.
    // if we have more than one cell which only one candidate digit, then RANDOMLY select a cell
    // (cells are identified by their index number)
    let mut change_made = false;
    let number_of_single_candidate_indices = single_candidate_indices.len();
    if number_of_single_candidate_indices > 0
    {
        // randomly pick a cell which has only one digit possibility
        // candidate is 0-8, representing digits 1-9
        let (board_single_candidate_index, digit) = get_random_single_candidate_cell(&mut rnglcg, &board_candidate_masks, single_candidate_indices);
        board[board_single_candidate_index].digit = digit;  // Set cell to the one digit it can be. Here's the +1 to convert to a digit 1-9

        board_candidate_masks[board_single_candidate_index] = 0; // clear candidates for this cell now that we've set cell to a digit
        change_made = true;  // we made a change to the state and board
        // message to user we set a particular cell to the only digit it could be
        let row = index_to_row(board_single_candidate_index);
        let col = index_to_col(board_single_candidate_index);
        let s = format!("({0}, {1}) can only contain {2}.", row + 1, col + 1, digit);
        log(&s);
    }
    change_made
}

// return a cell which can be set to only one number
fn get_random_single_candidate_cell(rnglcg: &mut PortableLCG, board_candidate_masks: &[u32; 81], single_candidate_cells: Vec<usize>) -> (usize, i32) {
    // randomly select one of the cells in single_candidate_cells list
    let random_single_candidate_index: usize = rnglcg.next_range(single_candidate_cells.len() as i32).try_into().unwrap();
    // single_candidate_index identifies the cell we are talking about
    let board_random_single_candidate_cell = single_candidate_cells[random_single_candidate_index];
    // candidate_mask is the one candidate digit we can use in this cell
    // candidate is the one digit we can use in this cell 0-8 (add one to get digit)
    let single_candidate_mask = board_candidate_masks[board_random_single_candidate_cell];  // candidate_mask has 1 bit set in range 0-8 indicating which digit 1-9 we can put in this cell
    let digit = get_single_bitmask_to_digit()[&(single_candidate_mask as usize)]; // digit 1-9
    (board_random_single_candidate_cell, digit)
}

// candidate_masks : list of 81, for each cell, the digits which are candidates (encoded as a bit mask)
fn get_single_candidate_indices(candidate_masks: &mut [u32; 81]) -> Vec<usize> {
    let single_candidate_indices: Vec<usize> = candidate_masks
        .iter()
        .enumerate()
        .filter_map(|(index, &mask)| {
            let candidates_count = mask_to_ones_count().get(&mask).copied().unwrap_or(0);
            if candidates_count == 1 {  // when there's only one digit that will work for this cell
                Some(index)
            } else {
                None
            }
        })
        .collect();
    single_candidate_indices
}

fn get_indices() -> BTreeMap<usize, Vec<Cell>> {
    let row_indices: BTreeMap<usize, Vec<Cell>> = (0..81)
        .map(|index| {
            let discriminator = index_to_row(index);
            (discriminator, Cell {
                description: format!("row #{}", discriminator + 1),
                index,
                digit: 0,
            })
        })
        .fold(BTreeMap::new(), |mut acc, (key, cell)| {
            acc.entry(key).or_default().push(cell);
            acc
        });
    // 31.
    // Group by columns
    // Create list of all 81 cells, grouped by COLUMN. Discriminator is COLUMN, varies from 9 - 17 (subtract 9 to get column)
    // Create the projected items using a for loop instead of Select
    let column_indices: BTreeMap<usize, Vec<Cell>> = (0..81)
        .map(|index| {
            let discriminator = 9 + index_to_col(index);
            (discriminator, Cell {
                description: format!("column #{}", index_to_col(index) + 1),
                index,
                digit: 0,
            })
        })
        .fold(BTreeMap::new(), |mut acc, (key, cell)| {
            acc.entry(key).or_default().push(cell);
            acc
        });
    // Group by blocks
    // Create list of all 81 cells, grouped by BLOCK. Discriminator is BLOCK, varies from 18-26 (subtract 18 to get BLOCK)
    // 32.
    let block_indices = (0..81).fold(BTreeMap::<usize, Vec<Cell>>::new(), |mut temp_map, index| {
        let discriminator = 18 + index_to_block_index(index);
        let cell = Cell {
            description: format!("block ({}, {})", index_to_block_row(index) + 1, index_to_block_col(index) + 1),
            index,
            digit: 0,
        };
        temp_map.entry(discriminator).or_insert_with(Vec::new).push(cell);
        temp_map
    });
    // Combine all groups
    // 33.
    let cell_groups: BTreeMap<usize, Vec<Cell>> = row_indices
        .into_iter()
        .chain(column_indices)
        .chain(block_indices)
        .collect();
    //#endregion
    cell_groups
}

fn calculate_candidates(board: &Board) -> [u32; 81] {
    //#region Calculate candidates for current state of the board
    let mut candidate_masks: [u32; 81] = [0; 81];

    // go through every cell of board, calculcate candicate numbers for each cell that does not already have a number in it.
    // candidate numbers are any digit 1-9 not already in that cell's row, column, and block
    // candidate numbers are stored as a bitmask where bits 0-8 represent digits 1-9
    for i in 0..board.cells.len()  // 28.
    {
        // if this cell doesn't already have a number in it,
        // then calculate all the numbers which can go in this cell
        // candidate_mask is bits 0-8 representing numbers 1-9
        // Array candidate_masks[81] is all candidate numbers for each cell of board
        if board[i].digit == 0
        {
            let row = index_to_row(i);
            let col = index_to_col(i);
            let block_row = index_to_block_row(i);
            let block_col = index_to_block_col(i);

            let mut colliding_numbers: u32 = 0;
            for j in 0..9
            {
                let row_sibling_index = 9 * row + j;
                let col_sibling_index = 9 * j + col;
                let block_sibling_index = 9 * (block_row * 3 + j / 3) + block_col * 3 + j % 3;
                // state[81] holds numbers for each cell (or 0 if no number set yet).
                let row_digit = if board[row_sibling_index].digit == 0 { 31 } else { board[row_sibling_index].digit };
                let col_digit = if board[col_sibling_index].digit == 0 { 31 } else { board[col_sibling_index].digit };
                let block_digit = if board[block_sibling_index].digit == 0 { 31 } else { board[block_sibling_index].digit };
                // mask has one bit set indicating the digit in state cell. bit 0 = 1, bit 1 = 2,... bit8 = 9. Only bits 0-8 are used.
                let row_sibling_mask: u32 = convert_digit_to_mask(row_digit);
                let col_sibling_mask: u32 = convert_digit_to_mask(col_digit);
                let block_sibling_mask: u32 = convert_digit_to_mask(block_digit);
                // colliding_numbers bits 0-8 indicate what numbers are already present in this cell's row, column, and block.
                // This is complete list of numbers already used. Stored in one integer.
                colliding_numbers = colliding_numbers | row_sibling_mask | col_sibling_mask | block_sibling_mask;
            }
            // candidate_mask is all numbers available to put in cell[i] of board (state)
            // all the not "already used" numbers
            let all_ones: u32 = (1 << 9) - 1;
            candidate_masks[i] = all_ones & !colliding_numbers; // mask indicating what numbers are available for this cell.
        }
    }
    //#endregion
    candidate_masks
}

fn convert_digit_to_mask(digit: i32) -> u32 {
    1 << (digit - 1)
}

// lazy calculate
//Dictionary<int, int> maskToOnesCount = new Dictionary<int, int>();
// Key is 0-511, value is number of binary bits in binary representation of key
// algorithm is, as we build the table temp_map[], consider 8 bit byte i, for any i,
// we first determine how many bits are set for byte bits 1-7, excluding the lowest bit.
// We shift i >> 1, which gives us i/2. We now look up the result for i/2 in table temp_map.
// Now we just need to add in the lowest bit, bit 0, to get the total number of bit set.
// smaller is i/2 (i >> 1), increment is lowest bit 0 (either 0 or 1)
// Number of bits set in value i is number of bits set in i/2 + lowest bit of i.
static MASK_TO_ONES_COUNT: OnceLock<BTreeMap<u32, usize>> = OnceLock::new();
fn mask_to_ones_count() -> &'static BTreeMap<u32, usize> {
    MASK_TO_ONES_COUNT.get_or_init(|| {
        let mut answers = BTreeMap::new();
        answers.insert(0, 0);
        for i in 1..(1 << 9) {                              // 1 << 9 = 512. Thus goes from 1 to 511
            let half: u32 = i >> 1;                                 // half = i >> 1 (i/2). Shift to a number we have already solved and stored in answers
            let lowbit: usize = (i & 1) as usize;                   // lowbit is lowest bit of i. Either 0 or 1
            let new_result = answers[&half] + lowbit;        // bits set in i is bits set in i >> 1 (answers[half]) + lowest bit of i (lowbit)
            answers.insert(i, new_result);                   // store the value. i, new_result=number of bits set in i
        }
        answers
    })
}

// lazy calculate single_bit_to_index
// key: 1  value: 0
// key: 2  value: 1
// key: 4  value: 2
// key: 8  value: 3
// key: 16  value: 4
// key: 32  value: 5
// key: 64  value: 6
// key: 128  value: 7
// key: 256  value: 8

static SINGLE_BIT_TO_INDEX: OnceLock<HashMap<usize, i32> > = OnceLock::new();
fn get_single_bitmask_to_digit() -> &'static HashMap<usize, i32> {
    SINGLE_BIT_TO_INDEX.get_or_init(|| {
        let mut single_bit_to_index: HashMap<usize, i32> = HashMap::new();
        for i in 0..9
        {
            single_bit_to_index.insert(1 << i, i+1);
        }
        single_bit_to_index
    })
}

// Convert mask to list of digits they represent. Returns list of digits.
fn mask_to_vec_digits(input_mask: u32) -> Vec<i32> {
    let mut mask_to_remove = input_mask;
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
    values_to_remove
}

fn top_two_digits(value: u32) -> (i32, i32) {
    let mut temp = value;
    let mut lower = 0;
    let mut upper = 0;
    let mut digit = 1;
    while temp > 0
    {
        if (temp & 1) != 0
        {
            lower = upper;
            upper = digit;
        }
        temp = temp >> 1;
        digit += 1;
    }
    (lower, upper)
}

fn get_row_col_block_used_digits(current_state: &Board, index: usize) -> [bool; 9] {
    // gather all digits used in cell's row, column, and block. output is_digit_used. input: current_state, row, col, block_row, block_col
    let is_digit_used = gather_digits(&current_state, index);
    is_digit_used
}

fn gather_digits(current_state: &Board, index: usize) -> [bool; 9]  {
    let row = index_to_row(index);
    let col = index_to_col(index);
    let block_row = row / 3;
    let block_col = col / 3;
    let mut is_digit_used: [bool; 9] = [false; 9];

    for i in 0..9  // 12.
    {
        let cell = current_state[9 * i + col].clone();
        let row_digit = cell.digit;
        if row_digit > 0
        {
            is_digit_used[row_digit as usize - 1] = true;
        }

        let cell = current_state[9 * row + i].clone();
        let col_digit = cell.digit;
        if col_digit > 0
        {
            is_digit_used[col_digit as usize - 1] = true;
        }

        let cell = current_state[(block_row * 3 + i / 3) * 9 + (block_col * 3 + i % 3)].clone();
        let block_digit = cell.digit;
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


// Mutable because we update seed at every call
// From "Numerical Recipies"
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
    // r is range 0 to r.
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

fn log(s : &String)
{
    println!("{}", s);
    write_from_function(s).expect("TODO: panic message");
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

    // MAIN LOOP
    for seed in 1..25
    {
        let my_rng = PortableLCG::new(seed);
        log(&format!("RUN {}", seed));
        play(my_rng);
    }

    log(&"THE END!".to_string());
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
