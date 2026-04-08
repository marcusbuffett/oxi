use std::collections::{HashMap, VecDeque};
use std::sync::mpsc::{self, Receiver, Sender, TryRecvError};
use std::sync::{Arc, Mutex};
use std::thread::{self, JoinHandle};
use std::time::{Duration, Instant};

use crate::metrics_renderer::{
    EvaluationName, EvaluationProgress, MetricState, MetricsRenderer, MetricsRendererEvaluation,
    MetricsRendererTraining, PredictionEntry, TrainingProgress,
};
use anyhow::Context;
use burn_train::Interrupter;
use crossterm::{
    cursor,
    event::{self, Event, KeyCode, KeyEvent, KeyModifiers},
    execute,
    terminal::{disable_raw_mode, enable_raw_mode, EnterAlternateScreen, LeaveAlternateScreen},
};
use ratatui::{
    backend::CrosstermBackend,
    layout::{Constraint, Direction, Layout, Margin, Rect},
    style::{Color, Modifier, Style},
    symbols,
    text::{Line, Span},
    widgets::{
        Axis, Bar, BarChart, BarGroup, Block, Borders, Chart, Clear, Dataset, Gauge, GraphType,
        LegendPosition, Paragraph, Wrap,
    },
    Frame, Terminal,
};

const MAX_DURATION_SAMPLES: usize = 200;
const DRAW_INTERVAL: Duration = Duration::from_millis(100);
const LOOP_SLEEP: Duration = Duration::from_millis(30);

#[derive(Clone, Debug)]
enum MetricSplit {
    Train,
    Valid,
    Test(String),
}

impl MetricSplit {
    fn label(&self) -> String {
        match self {
            MetricSplit::Train => "train".to_string(),
            MetricSplit::Valid => "valid".to_string(),
            MetricSplit::Test(name) => format!("test:{name}"),
        }
    }
}

#[derive(Clone, Copy, Debug)]
struct MetricSample {
    iteration: usize,
    value: f64,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum MetricGroupKind {
    Line,
    Bar,
}

#[derive(Clone, Debug)]
struct MetricSeriesLine {
    id: String,
    label: String,
    samples: Vec<MetricSample>,
    last_formatted: Option<String>,
    color: Color,
}

impl MetricSeriesLine {
    fn new(id: String, label: String, color: Color) -> Self {
        Self {
            id,
            label,
            samples: Vec::new(),
            last_formatted: None,
            color,
        }
    }

    fn push_sample(&mut self, iteration: usize, value: f64) {
        self.samples.push(MetricSample { iteration, value });
    }

    fn last_value(&self) -> Option<f64> {
        self.samples.last().map(|sample| sample.value)
    }
}

#[derive(Clone, Debug)]
struct PredictionDisplay {
    label: String,
    probability: f64,
}

#[derive(Clone, Debug, Default)]
struct PredictionGroupData {
    entries: Vec<PredictionDisplay>,
    formatted: String,
}

#[derive(Clone, Debug)]
struct MetricGroup {
    id: String,
    name: String,
    split: MetricSplit,
    series: Vec<MetricSeriesLine>,
    kind: MetricGroupKind,
    predictions: Option<PredictionGroupData>,
}

impl MetricGroup {
    fn new(id: String, name: String, split: MetricSplit) -> Self {
        Self {
            id,
            name,
            split,
            series: Vec::new(),
            kind: MetricGroupKind::Line,
            predictions: None,
        }
    }

    fn display_name(&self) -> String {
        let prefix = self.split.label();
        format!("{prefix}/{}", self.name)
    }

    fn add_series(&mut self, line: MetricSeriesLine) -> usize {
        self.series.push(line);
        self.series.len() - 1
    }

    fn set_predictions(&mut self, formatted: String, entries: Vec<PredictionDisplay>) {
        self.kind = MetricGroupKind::Bar;
        self.series.clear();
        self.predictions = Some(PredictionGroupData { entries, formatted });
    }
}

#[derive(Default)]
struct ProgressState {
    iteration: usize,
    total_iterations: usize,
    items_processed: usize,
    items_total: usize,
    epoch: usize,
    epoch_total: usize,
    start_time: Option<Instant>,
    last_timestamp: Option<Instant>,
    durations: VecDeque<f64>,
}

impl ProgressState {
    fn new(total_iterations: usize, total_items: usize) -> Self {
        Self {
            total_iterations,
            items_total: total_items,
            ..Self::default()
        }
    }

    fn update(
        &mut self,
        iteration: usize,
        items_processed: usize,
        items_total: usize,
        epoch: usize,
        epoch_total: usize,
        timestamp: Instant,
    ) {
        if self.start_time.is_none() {
            self.start_time = Some(timestamp);
        }

        if let Some(last) = self.last_timestamp {
            let elapsed = timestamp.saturating_duration_since(last).as_secs_f64();
            if elapsed.is_finite() && elapsed > 0.0 {
                self.durations.push_back(elapsed);
                if self.durations.len() > MAX_DURATION_SAMPLES {
                    self.durations.pop_front();
                }
            }
        }

        self.last_timestamp = Some(timestamp);
        self.iteration = iteration;
        self.items_processed = items_processed;
        self.items_total = items_total;
        self.epoch = epoch;
        self.epoch_total = epoch_total;
    }

    fn iteration_for_metric(&self) -> usize {
        self.iteration.saturating_add(1)
    }

    fn is_streaming(&self) -> bool {
        self.total_iterations == usize::MAX
    }

    fn ratio(&self) -> Option<f64> {
        if self.is_streaming() || self.total_iterations == 0 {
            return None;
        }
        Some((self.iteration as f64 / self.total_iterations as f64).clamp(0.0, 1.0))
    }

    fn elapsed(&self) -> Option<Duration> {
        let start = self.start_time?;
        let last = self.last_timestamp?;
        Some(last.saturating_duration_since(start))
    }
}

enum TuiMessage {
    Metric {
        split: MetricSplit,
        name: String,
        formatted: String,
        numeric: Option<f64>,
        predictions: Option<Vec<PredictionEntry>>,
    },
    Progress {
        iteration: usize,
        items_processed: usize,
        items_total: usize,
        epoch: usize,
        epoch_total: usize,
        timestamp: Instant,
    },
    Shutdown,
}

struct TuiApp {
    metric_groups: Vec<MetricGroup>,
    group_lookup: HashMap<String, usize>,
    series_lookup: HashMap<String, (usize, usize)>,
    display_order: Vec<usize>,
    selected: usize,
    expanded: bool,
    should_exit: bool,
    confirm_exit: bool,
    progress: ProgressState,
    interrupter: Interrupter,
    dirty: bool,
    stage_label: Option<String>,
}

impl TuiApp {
    fn new(interrupter: Interrupter, total_iterations: usize, total_items: usize) -> Self {
        Self {
            metric_groups: Vec::new(),
            group_lookup: HashMap::new(),
            series_lookup: HashMap::new(),
            display_order: Vec::new(),
            selected: 0,
            expanded: false,
            should_exit: false,
            confirm_exit: false,
            progress: ProgressState::new(total_iterations, total_items),
            interrupter,
            dirty: true,
            stage_label: None,
        }
    }

    fn handle_message(&mut self, message: TuiMessage) {
        match message {
            TuiMessage::Metric {
                split,
                name,
                formatted,
                numeric,
                predictions,
            } => self.handle_metric(split, name, formatted, numeric, predictions),
            TuiMessage::Progress {
                iteration,
                items_processed,
                items_total,
                epoch,
                epoch_total,
                timestamp,
            } => {
                self.progress.update(
                    iteration,
                    items_processed,
                    items_total,
                    epoch,
                    epoch_total,
                    timestamp,
                );
                self.dirty = true;
            }
            TuiMessage::Shutdown => {
                self.should_exit = true;
                self.dirty = true;
            }
        }
    }

    fn handle_metric(
        &mut self,
        split: MetricSplit,
        name: String,
        formatted: String,
        numeric: Option<f64>,
        predictions: Option<Vec<PredictionEntry>>,
    ) {
        let (base_name, maybe_series) = split_metric_name(&name);
        let series_label = maybe_series.unwrap_or_else(|| base_name.clone());
        let group_id = format!("{}/{}", split.label(), base_name);
        let series_id = format!("{group_id}|{series_label}");

        if base_name.eq_ignore_ascii_case("Stage") {
            self.stage_label = Some(formatted);
            self.dirty = true;
            return;
        }

        let group_index = if let Some(index) = self.group_lookup.get(&group_id) {
            *index
        } else {
            let index = self.metric_groups.len();
            self.group_lookup.insert(group_id.clone(), index);
            self.metric_groups.push(MetricGroup::new(
                group_id.clone(),
                base_name.clone(),
                split.clone(),
            ));
            self.rebuild_display_order();
            index
        };

        if let Some(predictions) = predictions {
            let entries = predictions
                .into_iter()
                .map(|entry| PredictionDisplay {
                    label: entry.label,
                    probability: entry.probability,
                })
                .collect::<Vec<_>>();

            self.series_lookup
                .retain(|_, (existing_group, _)| *existing_group != group_index);

            if let Some(group) = self.metric_groups.get_mut(group_index) {
                group.set_predictions(formatted, entries);
            }

            self.dirty = true;

            if self.selected >= self.display_count() {
                self.selected = self.display_count().saturating_sub(1);
            }
            return;
        }

        let series_index = if let Some(&(existing_group, series_idx)) =
            self.series_lookup.get(&series_id)
        {
            if existing_group == group_index {
                series_idx
            } else {
                let color = self
                    .metric_groups
                    .get(group_index)
                    .map(|group| assign_series_color(group, &series_label))
                    .unwrap_or(Color::White);
                let line = MetricSeriesLine::new(series_id.clone(), series_label.clone(), color);
                let idx = self.metric_groups[group_index].add_series(line);
                self.series_lookup
                    .insert(series_id.clone(), (group_index, idx));
                idx
            }
        } else {
            let color = assign_series_color(&self.metric_groups[group_index], &series_label);
            let line = MetricSeriesLine::new(series_id.clone(), series_label.clone(), color);
            let idx = self.metric_groups[group_index].add_series(line);
            self.series_lookup
                .insert(series_id.clone(), (group_index, idx));
            idx
        };

        if let Some(value) = numeric {
            let iteration = self.progress.iteration_for_metric();
            if let Some(series) = self
                .metric_groups
                .get_mut(group_index)
                .and_then(|group| group.series.get_mut(series_index))
            {
                series.push_sample(iteration, value);
                series.last_formatted = Some(formatted);
            }
        } else if let Some(series) = self
            .metric_groups
            .get_mut(group_index)
            .and_then(|group| group.series.get_mut(series_index))
        {
            series.last_formatted = Some(formatted);
        }

        self.dirty = true;

        if self.selected >= self.display_count() {
            self.selected = self.display_count().saturating_sub(1);
        }
    }

    fn rebuild_display_order(&mut self) {
        self.display_order = (0..self.metric_groups.len()).collect();
        self.display_order.sort_by_key(|&idx| {
            let group = &self.metric_groups[idx];
            metric_display_priority(&group.name, &group.split)
        });
    }

    /// Map from display position to internal group index.
    fn display_index(&self, display_pos: usize) -> usize {
        self.display_order
            .get(display_pos)
            .copied()
            .unwrap_or(display_pos)
    }

    fn display_count(&self) -> usize {
        self.display_order.len()
    }

    fn request_exit_confirmation(&mut self) {
        if !self.confirm_exit {
            self.confirm_exit = true;
            self.dirty = true;
        }
    }

    fn handle_key(&mut self, key: KeyEvent) {
        if self.confirm_exit {
            match key.code {
                KeyCode::Char('c') | KeyCode::Char('C') => {
                    self.interrupter.stop(None);
                    self.should_exit = true;
                    self.confirm_exit = false;
                }
                _ => {
                    self.confirm_exit = false;
                    self.dirty = true;
                }
            }
            return;
        }

        match key.code {
            KeyCode::Char('q') => {
                self.request_exit_confirmation();
            }
            KeyCode::Char('c') if key.modifiers.contains(KeyModifiers::CONTROL) => {
                self.request_exit_confirmation();
            }
            KeyCode::Esc => {
                if self.expanded {
                    self.expanded = false;
                    self.dirty = true;
                } else {
                    self.request_exit_confirmation();
                }
            }
            KeyCode::Left | KeyCode::Char('h') => {
                if !self.display_order.is_empty() && self.selected > 0 {
                    self.selected -= 1;
                    self.dirty = true;
                }
            }
            KeyCode::Right | KeyCode::Char('l') => {
                if self.selected + 1 < self.display_count() {
                    self.selected += 1;
                    self.dirty = true;
                }
            }
            KeyCode::Up | KeyCode::Char('k') => {
                if !self.display_order.is_empty() {
                    let cols = grid_columns(self.display_count());
                    if self.selected >= cols {
                        self.selected -= cols;
                        self.dirty = true;
                    }
                }
            }
            KeyCode::Down | KeyCode::Char('j') => {
                if !self.display_order.is_empty() {
                    let cols = grid_columns(self.display_count());
                    let next = self.selected + cols;
                    if next < self.display_count() {
                        self.selected = next;
                        self.dirty = true;
                    }
                }
            }
            KeyCode::Char(' ') => {
                if !self.display_order.is_empty() {
                    self.expanded = !self.expanded;
                    self.dirty = true;
                }
            }
            _ => {}
        }
    }

    fn draw(&mut self, frame: &mut Frame) {
        let area = frame.area();
        let progress_height = if area.height > 8 { 5 } else { 3 };

        let chunks = Layout::default()
            .direction(Direction::Vertical)
            .constraints([Constraint::Min(5), Constraint::Length(progress_height)])
            .split(area);

        if self.expanded {
            self.draw_expanded(frame, chunks[0]);
        } else {
            self.draw_grid(frame, chunks[0]);
        }

        self.draw_progress(frame, chunks[1]);

        if self.confirm_exit {
            self.draw_exit_confirmation(frame);
        }
    }

    fn draw_grid(&self, frame: &mut Frame, area: Rect) {
        if self.display_order.is_empty() {
            let block = Block::default()
                .title("Metrics")
                .borders(Borders::ALL)
                .border_style(Style::default().fg(Color::DarkGray));
            frame.render_widget(block, area);

            let inner = area.inner(Margin {
                vertical: 1,
                horizontal: 1,
            });
            let placeholder = Paragraph::new("Waiting for metrics...")
                .style(Style::default().fg(Color::Gray))
                .alignment(ratatui::layout::Alignment::Center);
            frame.render_widget(placeholder, inner);
            return;
        }

        let count = self.display_count();
        let cols = grid_columns(count);
        let rows = (count + cols - 1) / cols;

        let row_constraints = (0..rows)
            .map(|_| Constraint::Ratio(1, rows as u32))
            .collect::<Vec<_>>();
        let row_chunks = Layout::default()
            .direction(Direction::Vertical)
            .constraints(row_constraints)
            .split(area);

        let mut display_pos = 0;
        for &row_area in row_chunks.iter() {
            if display_pos >= count {
                break;
            }
            let col_constraints = (0..cols)
                .map(|_| Constraint::Ratio(1, cols as u32))
                .collect::<Vec<_>>();
            let col_chunks = Layout::default()
                .direction(Direction::Horizontal)
                .constraints(col_constraints)
                .split(row_area);

            for &col_area in col_chunks.iter() {
                if display_pos >= count {
                    break;
                }
                let group_idx = self.display_index(display_pos);
                let is_selected = display_pos == self.selected;
                self.draw_chart(frame, col_area, group_idx, false, is_selected);
                display_pos += 1;
            }
        }
    }

    fn draw_expanded(&self, frame: &mut Frame, area: Rect) {
        if self.display_order.is_empty() {
            self.draw_grid(frame, area);
            return;
        }

        let main_width = if area.width > 20 {
            Constraint::Ratio(2, 3)
        } else {
            Constraint::Ratio(1, 1)
        };
        let details_width = if area.width > 20 {
            Constraint::Ratio(1, 3)
        } else {
            Constraint::Length(0)
        };

        let chunks = Layout::default()
            .direction(Direction::Horizontal)
            .constraints([main_width, details_width])
            .split(area);

        let group_idx = self.display_index(self.selected);
        self.draw_chart(frame, chunks[0], group_idx, true, true);

        if chunks.len() > 1 && chunks[1].width > 0 {
            self.draw_details(frame, chunks[1], group_idx);
        }
    }

    fn draw_chart(
        &self,
        frame: &mut Frame,
        area: Rect,
        index: usize,
        expanded: bool,
        is_selected: bool,
    ) {
        if index >= self.metric_groups.len() {
            return;
        }
        match self.metric_groups[index].kind {
            MetricGroupKind::Line => {
                self.draw_line_chart(frame, area, index, expanded, is_selected)
            }
            MetricGroupKind::Bar => {
                if self.metric_groups[index].predictions.is_some() {
                    self.draw_prediction_bars(frame, area, index, expanded, is_selected);
                } else {
                    self.draw_bar_chart(frame, area, index, expanded, is_selected);
                }
            }
        }
    }

    fn draw_line_chart(
        &self,
        frame: &mut Frame,
        area: Rect,
        index: usize,
        expanded: bool,
        is_selected: bool,
    ) {
        if index >= self.metric_groups.len() {
            return;
        }

        let group = &self.metric_groups[index];
        let usable_width = area.width.saturating_sub(4).max(1) as usize;

        let mut aggregated_data = Vec::with_capacity(group.series.len());
        let mut has_data = false;
        let mut bounds_initialized = false;
        let mut x_bounds = [0.0, 1.0];
        let mut y_bounds = [0.0, 1.0];

        for line in &group.series {
            let data = aggregate_samples(&line.samples, usable_width);
            if !data.is_empty() {
                let (series_x, series_y) = compute_bounds(&data);
                if !bounds_initialized {
                    x_bounds = series_x;
                    y_bounds = series_y;
                    bounds_initialized = true;
                } else {
                    x_bounds[0] = x_bounds[0].min(series_x[0]);
                    x_bounds[1] = x_bounds[1].max(series_x[1]);
                    y_bounds[0] = y_bounds[0].min(series_y[0]);
                    y_bounds[1] = y_bounds[1].max(series_y[1]);
                }
                has_data = true;
            }
            aggregated_data.push(data);
        }

        let mut block = Block::default()
            .title(group.display_name())
            .borders(Borders::ALL);

        if is_selected {
            block = block.border_style(
                Style::default()
                    .fg(if expanded { Color::Yellow } else { Color::Cyan })
                    .add_modifier(Modifier::BOLD),
            );
        } else {
            block = block.border_style(Style::default().fg(Color::DarkGray));
        }

        if !has_data {
            frame.render_widget(block.clone(), area);
            let inner = area.inner(Margin {
                vertical: 1,
                horizontal: 1,
            });
            let message = Paragraph::new("No data yet")
                .style(Style::default().fg(Color::Gray))
                .alignment(ratatui::layout::Alignment::Center);
            frame.render_widget(message, inner);
            return;
        }

        let legend_names: Vec<String> = group
            .series
            .iter()
            .map(|line| {
                if let Some(formatted) = &line.last_formatted {
                    format!("{}: {}", line.label, formatted)
                } else {
                    line.label.clone()
                }
            })
            .collect();

        let datasets: Vec<Dataset> = group
            .series
            .iter()
            .enumerate()
            .map(|(i, line)| {
                Dataset::default()
                    .name(legend_names[i].as_str())
                    .marker(symbols::Marker::Braille)
                    .graph_type(GraphType::Line)
                    .style(Style::default().fg(line.color))
                    .data(&aggregated_data[i])
            })
            .collect();

        let x_axis = Axis::default()
            .style(Style::default().fg(Color::Gray))
            .bounds(x_bounds);

        let y_axis = Axis::default()
            .style(Style::default().fg(Color::Gray))
            .bounds(y_bounds);

        let chart = Chart::new(datasets)
            .block(block)
            .x_axis(x_axis)
            .y_axis(y_axis)
            .legend_position(Some(LegendPosition::TopLeft))
            .hidden_legend_constraints((Constraint::Min(0), Constraint::Min(0)));

        frame.render_widget(chart, area);
    }

    fn draw_bar_chart(
        &self,
        frame: &mut Frame,
        area: Rect,
        index: usize,
        expanded: bool,
        is_selected: bool,
    ) {
        if index >= self.metric_groups.len() {
            return;
        }

        let group = &self.metric_groups[index];
        let mut block = Block::default()
            .title(group.display_name())
            .borders(Borders::ALL);

        if is_selected {
            block = block.border_style(
                Style::default()
                    .fg(if expanded { Color::Yellow } else { Color::Cyan })
                    .add_modifier(Modifier::BOLD),
            );
        } else {
            block = block.border_style(Style::default().fg(Color::DarkGray));
        }

        let mut bars = Vec::new();
        for line in &group.series {
            let raw_value = line.last_value().unwrap_or(0.0);
            let clamped = raw_value.clamp(0.0, 1.0);
            let percent = (clamped * 100.0).clamp(0.0, 100.0);
            let scaled = (percent * 10.0).round() as u64;
            let (label_text, display_percent) = parse_bar_metadata(line.last_formatted.as_ref());
            let label = if !label_text.is_empty() {
                label_text
            } else if line.label.len() > 16 {
                let mut truncated = line.label.chars().take(15).collect::<String>();
                truncated.push('…');
                truncated
            } else {
                line.label.clone()
            };
            let percent_display = display_percent.unwrap_or(percent as f64);
            let bar = Bar::default()
                .label(Line::from(label))
                .value(scaled)
                .text_value(format!("{percent_display:.2}%"))
                .style(Style::default().fg(line.color));
            bars.push(bar);
        }

        if group.series.iter().all(|line| line.samples.is_empty()) {
            frame.render_widget(block.clone(), area);
            let inner = area.inner(Margin {
                vertical: 1,
                horizontal: 1,
            });
            let message = Paragraph::new("No predictions yet")
                .style(Style::default().fg(Color::Gray))
                .alignment(ratatui::layout::Alignment::Center);
            frame.render_widget(message, inner);
            return;
        }

        let bar_count = bars.len() as u16;
        let bar_width = if bar_count == 0 {
            1
        } else {
            (area.width.saturating_sub(4) / bar_count.max(1))
                .max(1)
                .min(12)
        };

        let bar_group = BarGroup::default().bars(&bars);
        let chart = BarChart::default()
            .block(block)
            .bar_width(bar_width)
            .bar_gap(1)
            .group_gap(2)
            .max(1000)
            .value_style(Style::default().fg(Color::Yellow))
            .label_style(Style::default().fg(Color::Gray))
            .data(bar_group);

        frame.render_widget(chart, area);
    }

    fn draw_prediction_bars(
        &self,
        frame: &mut Frame,
        area: Rect,
        index: usize,
        expanded: bool,
        is_selected: bool,
    ) {
        if index >= self.metric_groups.len() {
            return;
        }

        let group = &self.metric_groups[index];
        let mut block = Block::default()
            .title(group.display_name())
            .borders(Borders::ALL);

        if is_selected {
            block = block.border_style(
                Style::default()
                    .fg(if expanded { Color::Yellow } else { Color::Cyan })
                    .add_modifier(Modifier::BOLD),
            );
        } else {
            block = block.border_style(Style::default().fg(Color::DarkGray));
        }

        let Some(data) = group.predictions.as_ref() else {
            frame.render_widget(block.clone(), area);
            let inner = area.inner(Margin {
                vertical: 1,
                horizontal: 1,
            });
            let message = Paragraph::new("No predictions yet")
                .style(Style::default().fg(Color::Gray))
                .alignment(ratatui::layout::Alignment::Center);
            frame.render_widget(message, inner);
            return;
        };

        if data.entries.is_empty() {
            frame.render_widget(block.clone(), area);
            let inner = area.inner(Margin {
                vertical: 1,
                horizontal: 1,
            });
            let message = Paragraph::new("No predictions yet")
                .style(Style::default().fg(Color::Gray))
                .alignment(ratatui::layout::Alignment::Center);
            frame.render_widget(message, inner);
            return;
        }

        let bars: Vec<Bar> = data
            .entries
            .iter()
            .enumerate()
            .map(|(idx, entry)| {
                let percentile = (entry.probability * 100.0).clamp(0.0, 100.0);
                let value = (percentile * 10.0).round() as u64;
                let mut label = entry.label.clone();
                if label.chars().count() > 16 {
                    label = label.chars().take(15).collect();
                    label.push('…');
                }
                let style = match idx {
                    0 => Style::default()
                        .fg(Color::Green)
                        .add_modifier(Modifier::BOLD),
                    1 => Style::default().fg(Color::Yellow),
                    _ => Style::default().fg(Color::Cyan),
                };
                Bar::default()
                    .label(Line::from(label))
                    .value(value)
                    .text_value(format!("{percentile:.2}%"))
                    .style(style)
            })
            .collect();

        let count = bars.len().max(1) as u16;
        let bar_width = (area.width.saturating_sub(4) / count).max(1).min(12);

        let bar_group = BarGroup::default().bars(&bars);
        let chart = BarChart::default()
            .block(block)
            .bar_width(bar_width)
            .bar_gap(1)
            .group_gap(2)
            .max(1000)
            .value_style(Style::default().fg(Color::Yellow))
            .label_style(Style::default().fg(Color::Gray))
            .data(bar_group);

        frame.render_widget(chart, area);
    }

    fn draw_details(&self, frame: &mut Frame, area: Rect, index: usize) {
        if index >= self.metric_groups.len() {
            return;
        }

        let group = &self.metric_groups[index];

        if group.kind == MetricGroupKind::Bar && group.predictions.is_some() {
            self.draw_prediction_details(frame, area, group);
            return;
        }

        let value_style = Style::default()
            .fg(Color::White)
            .add_modifier(Modifier::BOLD);

        let mut stats_lines = Vec::new();
        for line in group.series.iter() {
            let header_style = Style::default().fg(line.color).add_modifier(Modifier::BOLD);

            let latest_text = format_value(line.last_value());
            let avg_10 = average_last(&line.samples, 10);
            let avg_100 = average_last(&line.samples, 100);
            let prev_100 = average_previous(&line.samples, 100);

            let mut spans = vec![
                Span::styled(line.label.as_str(), header_style),
                Span::raw(": Latest "),
                Span::styled(latest_text, value_style),
                Span::raw(" | Avg10 "),
                Span::styled(format_value(avg_10), value_style),
                Span::raw(" | Avg100 "),
                Span::styled(format_value(avg_100), value_style),
            ];

            if let (Some(current), Some(previous)) = (avg_100, prev_100) {
                let (delta_text, delta_color) = format_delta(current - previous, Some(previous));
                let delta_style = Style::default()
                    .fg(delta_color)
                    .add_modifier(Modifier::BOLD);
                spans.push(Span::raw(" | Δ100 "));
                spans.push(Span::styled(delta_text, delta_style));
            }

            stats_lines.push(Line::from(spans));
        }

        let mut stats_height = group.series.len() as u16 + 2;
        stats_height = stats_height.max(3);
        stats_height = stats_height.min(area.height.max(3));

        let layout = Layout::default()
            .direction(Direction::Vertical)
            .constraints([Constraint::Length(stats_height), Constraint::Min(0)])
            .split(area);

        let stats = Paragraph::new(stats_lines)
            .block(
                Block::default()
                    .title("Statistics")
                    .borders(Borders::ALL)
                    .border_style(Style::default().fg(Color::DarkGray)),
            )
            .alignment(ratatui::layout::Alignment::Left);

        frame.render_widget(stats, layout[0]);

        if layout.len() > 1 && layout[1].height > 0 {
            let formatted_details = group
                .series
                .iter()
                .filter_map(|line| {
                    line.last_formatted
                        .as_ref()
                        .map(|value| format!("{}: {value}", line.label))
                })
                .collect::<Vec<_>>();

            let formatted = if formatted_details.is_empty() {
                "No formatted output available.".to_string()
            } else {
                formatted_details.join("\n\n")
            };

            let details = Paragraph::new(formatted)
                .wrap(Wrap { trim: true })
                .block(
                    Block::default()
                        .title("Latest Updates")
                        .borders(Borders::ALL)
                        .border_style(Style::default().fg(Color::DarkGray)),
                )
                .style(Style::default().fg(Color::Gray));
            frame.render_widget(details, layout[1]);
        }
    }

    fn draw_prediction_details(&self, frame: &mut Frame, area: Rect, group: &MetricGroup) {
        let block = Block::default()
            .title("Details")
            .borders(Borders::ALL)
            .border_style(Style::default().fg(Color::DarkGray));
        frame.render_widget(block.clone(), area);

        let inner = area.inner(Margin {
            vertical: 1,
            horizontal: 1,
        });

        let Some(data) = group.predictions.as_ref() else {
            let message = Paragraph::new("No predictions yet")
                .style(Style::default().fg(Color::Gray))
                .alignment(ratatui::layout::Alignment::Center);
            frame.render_widget(message, inner);
            return;
        };

        if data.entries.is_empty() {
            let message = Paragraph::new("No predictions yet")
                .style(Style::default().fg(Color::Gray))
                .alignment(ratatui::layout::Alignment::Center);
            frame.render_widget(message, inner);
            return;
        }

        let mut lines = Vec::new();
        for (idx, entry) in data.entries.iter().enumerate() {
            let rank = format!("{:>2}.", idx + 1);
            let style = if idx == 0 {
                Style::default()
                    .fg(Color::Green)
                    .add_modifier(Modifier::BOLD)
            } else {
                Style::default().fg(Color::White)
            };
            lines.push(Line::from(vec![
                Span::styled(rank, Style::default().fg(Color::Gray)),
                Span::raw(" "),
                Span::styled(entry.label.clone(), style),
                Span::raw(" "),
                Span::styled(
                    format!("{:.2}%", entry.probability * 100.0),
                    Style::default().fg(Color::Yellow),
                ),
            ]));
        }

        if !data.formatted.is_empty() {
            lines.push(Line::from(""));
            lines.push(Line::from(vec![Span::styled(
                data.formatted.clone(),
                Style::default().fg(Color::Gray),
            )]));
        }

        let paragraph = Paragraph::new(lines)
            .wrap(Wrap { trim: false })
            .alignment(ratatui::layout::Alignment::Left);

        frame.render_widget(paragraph, inner);
    }

    fn draw_progress(&self, frame: &mut Frame, area: Rect) {
        let is_streaming = self.progress.is_streaming();

        let (gauge_area, info_area) = if is_streaming {
            (None, area)
        } else {
            let chunks = Layout::default()
                .direction(Direction::Horizontal)
                .constraints([Constraint::Ratio(3, 4), Constraint::Ratio(1, 4)])
                .split(area);
            (Some(chunks[0]), chunks[1])
        };

        if let Some(gauge_area) = gauge_area {
            let ratio = self.progress.ratio().unwrap_or(0.0);
            let label = format!("{:.1}%", ratio * 100.0);

            let gauge = Gauge::default()
                .block(
                    Block::default()
                        .title("Training Progress")
                        .borders(Borders::ALL)
                        .border_style(Style::default().fg(Color::DarkGray)),
                )
                .gauge_style(
                    Style::default()
                        .fg(Color::Green)
                        .bg(Color::Black)
                        .add_modifier(Modifier::BOLD),
                )
                .ratio(ratio)
                .label(label);

            frame.render_widget(gauge, gauge_area);
        }

        let iterations_text = if is_streaming {
            format!("Iteration: {}", self.progress.iteration)
        } else {
            format!(
                "{} / {} iterations",
                self.progress.iteration, self.progress.total_iterations
            )
        };

        let elapsed_text = self
            .progress
            .elapsed()
            .map(format_duration)
            .unwrap_or_else(|| "--".to_string());

        let mut info_lines = vec![
            Line::from(Span::styled(
                iterations_text,
                Style::default().fg(Color::White),
            )),
            Line::from(vec![
                Span::styled("Elapsed: ", Style::default().fg(Color::Gray)),
                Span::styled(elapsed_text, Style::default().fg(Color::Yellow)),
            ]),
        ];

        if let Some(stage) = &self.stage_label {
            info_lines.push(Line::from(vec![
                Span::styled("Stage: ", Style::default().fg(Color::Gray)),
                Span::styled(stage.as_str(), Style::default().fg(Color::Cyan)),
            ]));
        }

        let title = if is_streaming {
            "Training (Streaming)"
        } else {
            "Progress Info"
        };

        let info = Paragraph::new(info_lines)
            .block(
                Block::default()
                    .borders(Borders::ALL)
                    .border_style(Style::default().fg(Color::DarkGray))
                    .title(title),
            )
            .alignment(ratatui::layout::Alignment::Left);

        frame.render_widget(info, info_area);
    }

    fn draw_exit_confirmation(&self, frame: &mut Frame) {
        let area = frame.area();
        if area.width == 0 || area.height == 0 {
            return;
        }

        let popup_width = if area.width <= 28 {
            area.width
        } else {
            area.width.min(48)
        };
        let popup_height = if area.height <= 7 {
            area.height
        } else {
            area.height.min(9)
        };

        let popup_x = area.x + (area.width - popup_width) / 2;
        let popup_y = area.y + (area.height - popup_height) / 2;
        let popup_area = Rect::new(popup_x, popup_y, popup_width, popup_height);

        frame.render_widget(Clear, popup_area);

        let lines = vec![
            Line::from(Span::styled(
                "Quit training and close the dashboard?",
                Style::default()
                    .fg(Color::White)
                    .add_modifier(Modifier::BOLD),
            )),
            Line::from("Press c to confirm. Any other key cancels."),
        ];

        let prompt = Paragraph::new(lines)
            .alignment(ratatui::layout::Alignment::Center)
            .wrap(Wrap { trim: true })
            .block(
                Block::default()
                    .title("Confirm Exit")
                    .borders(Borders::ALL)
                    .border_style(Style::default().fg(Color::Yellow))
                    .style(Style::default().bg(Color::Black)),
            )
            .style(Style::default().fg(Color::White));

        frame.render_widget(prompt, popup_area);
    }

    fn take_dirty(&mut self) -> bool {
        std::mem::take(&mut self.dirty)
    }
}

fn grid_columns(count: usize) -> usize {
    if count <= 1 {
        return 1;
    }
    let cols = (count as f64).sqrt().ceil() as usize;
    cols.max(1)
}

/// Returns a sort key for metric display order.
/// Lower values appear first. Groups:
///   0xx - Losses (total, policy, value, aux)
///   1xx - Accuracy metrics (top-1, top-5, WDL, puzzle, elo/stage breakdowns)
///   2xx - Calibration metrics
///   3xx - Training diagnostics (gradient norm, learning rate, iteration speed)
///   9xx - Unknown / other
fn metric_display_priority(base_name: &str, split: &MetricSplit) -> u32 {
    // Split prefix: train < valid < test
    let split_offset = match split {
        MetricSplit::Train => 0,
        MetricSplit::Valid => 10000,
        MetricSplit::Test(_) => 20000,
    };

    let name_lower = base_name.to_lowercase();

    let priority = if name_lower == "loss" || name_lower == "total loss" {
        0
    } else if name_lower.contains("policy") && name_lower.contains("loss") {
        10
    } else if name_lower.contains("value") && name_lower.contains("loss") {
        20
    } else if name_lower == "aux loss" {
        30
    } else if name_lower.contains("aux") && name_lower.contains("loss") {
        40
    } else if name_lower.contains("move top-1 accuracy by elo") {
        110
    } else if name_lower.contains("move top-1 accuracy by game stage") {
        120
    } else if name_lower.contains("move top-1") || name_lower.contains("top-1") {
        100
    } else if name_lower.contains("move top-5") || name_lower.contains("top-5") {
        130
    } else if name_lower.contains("wdl") {
        140
    } else if name_lower.contains("puzzle") {
        150
    } else if name_lower.contains("aux") && name_lower.contains("accuracy") {
        160
    } else if name_lower.contains("aux") && name_lower.contains("mae") {
        170
    } else if name_lower.contains("cp loss calibration by elo") || name_lower.contains("centipawn") && name_lower.contains("by elo") {
        210
    } else if name_lower.contains("centipawn") || name_lower.contains("calibration") {
        200
    } else if name_lower.contains("gradient") || name_lower.contains("grad norm") {
        300
    } else if name_lower.contains("learning rate") || name_lower == "lr" {
        310
    } else if name_lower.contains("iteration speed") {
        320
    } else {
        900
    };

    split_offset + priority
}

fn split_metric_name(name: &str) -> (String, Option<String>) {
    if let Some((base, series)) = name.split_once('|') {
        (base.trim().to_string(), Some(series.trim().to_string()))
    } else {
        (name.to_string(), None)
    }
}

const COLOR_PALETTE: [Color; 10] = [
    Color::Cyan,
    Color::LightCyan,
    Color::Yellow,
    Color::LightYellow,
    Color::Green,
    Color::LightGreen,
    Color::Magenta,
    Color::LightMagenta,
    Color::Blue,
    Color::LightBlue,
];

/// Fixed colors for well-known series labels (elo bands, game stages, etc.)
fn fixed_series_color(label: &str) -> Option<Color> {
    match label {
        // Elo bands - consistent warm-to-cool gradient
        "Beginner" => Some(Color::Green),
        "Intermediate" => Some(Color::Yellow),
        "Advanced" => Some(Color::Cyan),
        "Expert" => Some(Color::Magenta),
        // Game stages
        "Opening" => Some(Color::LightCyan),
        "Middlegame" => Some(Color::Yellow),
        "Endgame" => Some(Color::LightMagenta),
        // Aux heads
        "Mobility" => Some(Color::Cyan),
        "Material" => Some(Color::Yellow),
        // Aux square heads
        "From Sq" => Some(Color::LightCyan),
        "To Sq" => Some(Color::LightYellow),
        "Side Info" => Some(Color::LightGreen),
        // Calibration
        "Policy MAE" => Some(Color::Cyan),
        "Head MAE" => Some(Color::Yellow),
        "Head CE" => Some(Color::Magenta),
        "Labeled Fraction" => Some(Color::Green),
        "Overall" => Some(Color::LightCyan),
        _ => None,
    }
}

fn assign_series_color(group: &MetricGroup, label: &str) -> Color {
    if label.is_empty() {
        return Color::White;
    }
    if let Some(color) = fixed_series_color(label) {
        return color;
    }
    let mut index = (stable_hash(label) % COLOR_PALETTE.len() as u64) as usize;
    for _ in 0..COLOR_PALETTE.len() {
        let candidate = COLOR_PALETTE[index];
        if group.series.iter().all(|line| line.color != candidate) {
            return candidate;
        }
        index = (index + 1) % COLOR_PALETTE.len();
    }
    Color::White
}

fn stable_hash(value: &str) -> u64 {
    const OFFSET: u64 = 0xcbf29ce484222325;
    const PRIME: u64 = 0x100000001b3;
    let mut hash = OFFSET;
    for byte in value.as_bytes() {
        hash ^= *byte as u64;
        hash = hash.wrapping_mul(PRIME);
    }
    hash
}

fn parse_bar_metadata(formatted: Option<&String>) -> (String, Option<f64>) {
    if let Some(text) = formatted {
        if let Some((label, value)) = text.split_once("|") {
            let label = label.trim().to_string();
            let value = value.trim().parse::<f64>().ok();
            return (label, value);
        }
    }
    (String::new(), None)
}

fn aggregate_samples(samples: &[MetricSample], max_points: usize) -> Vec<(f64, f64)> {
    if samples.is_empty() || max_points == 0 {
        return Vec::new();
    }

    if samples.len() <= max_points {
        return samples
            .iter()
            .map(|sample| (sample.iteration as f64, sample.value))
            .collect();
    }

    let chunk_size = ((samples.len() as f64) / (max_points as f64)).ceil() as usize;
    if chunk_size == 0 {
        return samples
            .iter()
            .map(|sample| (sample.iteration as f64, sample.value))
            .collect();
    }

    samples
        .chunks(chunk_size)
        .map(|chunk| {
            let len = chunk.len() as f64;
            let avg_iter = chunk.iter().map(|s| s.iteration as f64).sum::<f64>() / len;
            let avg_value = chunk.iter().map(|s| s.value).sum::<f64>() / len;
            (avg_iter, avg_value)
        })
        .collect()
}

fn compute_bounds(data: &[(f64, f64)]) -> ([f64; 2], [f64; 2]) {
    if data.is_empty() {
        return ([0.0, 1.0], [0.0, 1.0]);
    }

    let mut min_x = f64::MAX;
    let mut max_x = f64::MIN;
    let mut min_y = f64::MAX;
    let mut max_y = f64::MIN;

    for (x, y) in data {
        min_x = min_x.min(*x);
        max_x = max_x.max(*x);
        min_y = min_y.min(*y);
        max_y = max_y.max(*y);
    }

    if (max_x - min_x).abs() < f64::EPSILON {
        max_x = min_x + 1.0;
    }

    if (max_y - min_y).abs() < f64::EPSILON {
        let padding = (min_y.abs().max(1.0)) * 0.05 + 1e-6;
        min_y -= padding;
        max_y += padding;
    }

    ([min_x, max_x], [min_y, max_y])
}

fn average_last(samples: &[MetricSample], count: usize) -> Option<f64> {
    if count == 0 || samples.len() < count {
        return None;
    }
    let slice = &samples[samples.len() - count..];
    Some(slice.iter().map(|s| s.value).sum::<f64>() / slice.len() as f64)
}

fn average_previous(samples: &[MetricSample], count: usize) -> Option<f64> {
    if count == 0 || samples.len() < count * 2 {
        return None;
    }
    let start = samples.len() - count * 2;
    let slice = &samples[start..start + count];
    Some(slice.iter().map(|s| s.value).sum::<f64>() / slice.len() as f64)
}

fn format_value(value: Option<f64>) -> String {
    match value {
        Some(v) if v.is_finite() => {
            if v < 0.01 && v != 0.0 {
                format!("{:.2e}", v)
            } else {
                format!("{:.4}", v)
            }
        }
        _ => "--".to_string(),
    }
}

fn format_delta(delta: f64, previous: Option<f64>) -> (String, Color) {
    if !delta.is_finite() {
        return ("--".to_string(), Color::Gray);
    }

    let sign = if delta >= 0.0 { "+" } else { "" };
    let color = if delta > 0.0 {
        Color::Green
    } else if delta < 0.0 {
        Color::Red
    } else {
        Color::Gray
    };

    let delta_str = if delta.abs() < 0.01 && delta.abs() > f64::EPSILON {
        format!("{:.2e}", delta.abs())
    } else {
        format!("{:.4}", delta.abs())
    };

    let text = if let Some(prev) = previous {
        if prev.abs() > f64::EPSILON {
            let percent = (delta / prev.abs()) * 100.0;
            format!("{}{} ({}{:.1}%)", sign, delta_str, sign, percent)
        } else {
            format!("{}{}", sign, delta_str)
        }
    } else {
        format!("{}{}", sign, delta_str)
    };

    (text, color)
}

fn format_duration(duration: Duration) -> String {
    if duration.is_zero() {
        return "0s".to_string();
    }
    let secs = duration.as_secs();
    if secs >= 3600 {
        let hours = secs / 3600;
        let minutes = (secs % 3600) / 60;
        let seconds = secs % 60;
        format!("{hours:02}:{minutes:02}:{seconds:02}")
    } else if secs >= 60 {
        let minutes = secs / 60;
        let seconds = secs % 60;
        format!("{minutes:02}:{seconds:02}")
    } else if secs >= 1 {
        format!("{:02}s", secs)
    } else {
        format!("{:.1}s", duration.as_secs_f64())
    }
}

struct TerminalCleanup;

impl Drop for TerminalCleanup {
    fn drop(&mut self) {
        let _ = disable_raw_mode();
        let mut stdout = std::io::stdout();
        let _ = execute!(stdout, LeaveAlternateScreen, cursor::Show);
    }
}

fn run_tui_thread(
    rx: Receiver<TuiMessage>,
    interrupter: Interrupter,
    total_iterations: usize,
    total_items: usize,
) -> anyhow::Result<()> {
    let mut stdout = std::io::stdout();
    execute!(stdout, EnterAlternateScreen, cursor::Hide).context("enter alternate screen")?;
    enable_raw_mode().context("enable raw mode")?;
    let backend = CrosstermBackend::new(stdout);
    let mut terminal = Terminal::new(backend).context("create terminal")?;
    terminal.hide_cursor().ok();

    let _cleanup = TerminalCleanup;
    let mut app = TuiApp::new(interrupter, total_iterations, total_items);
    let mut last_draw = Instant::now();
    let mut exit_requested = false;

    while !exit_requested {
        let mut drained = false;
        loop {
            match rx.try_recv() {
                Ok(message) => match message {
                    TuiMessage::Shutdown => {
                        exit_requested = true;
                        break;
                    }
                    other => {
                        app.handle_message(other);
                        drained = true;
                    }
                },
                Err(TryRecvError::Empty) => break,
                Err(TryRecvError::Disconnected) => {
                    exit_requested = true;
                    break;
                }
            }
        }

        while !exit_requested && event::poll(Duration::from_millis(0)).unwrap_or(false) {
            match event::read() {
                Ok(Event::Key(key)) => {
                    app.handle_key(key);
                }
                Ok(Event::Resize(_, _)) => {
                    app.dirty = true;
                }
                Ok(_) => {}
                Err(_) => {
                    exit_requested = true;
                    break;
                }
            }
        }

        if app.should_exit {
            exit_requested = true;
        }

        let dirty = app.take_dirty();
        if dirty || drained || last_draw.elapsed() >= DRAW_INTERVAL {
            terminal
                .draw(|frame| app.draw(frame))
                .context("draw frame")?;
            last_draw = Instant::now();
        }

        if exit_requested {
            break;
        }

        thread::sleep(LOOP_SLEEP);
    }

    terminal.show_cursor().ok();
    Ok(())
}

pub struct OxiTuiRenderer {
    tx: Arc<Mutex<Sender<TuiMessage>>>,
    handle: Option<JoinHandle<()>>,
    interrupter: Interrupter,
}

impl OxiTuiRenderer {
    pub fn new(interruptor: Interrupter, total_iterations: usize, total_items: usize) -> Self {
        let (tx, rx) = mpsc::channel::<TuiMessage>();
        let tx_arc = Arc::new(Mutex::new(tx));
        let interrupter_clone = interruptor.clone();

        let handle = thread::Builder::new()
            .name("oxi-tui".to_string())
            .spawn(move || {
                if let Err(err) =
                    run_tui_thread(rx, interrupter_clone, total_iterations, total_items)
                {
                    eprintln!("TUI thread terminated: {err:?}");
                }
            })
            .expect("failed to spawn TUI thread");

        Self {
            tx: tx_arc,
            handle: Some(handle),
            interrupter: interruptor,
        }
    }

    fn send(&self, message: TuiMessage) {
        if let Ok(sender) = self.tx.lock() {
            let _ = sender.send(message);
        }
    }

    fn forward_metric(&mut self, split: MetricSplit, state: MetricState) {
        match state {
            MetricState::Generic { name, entry } => {
                self.send(TuiMessage::Metric {
                    split,
                    name,
                    formatted: entry.formatted.clone(),
                    numeric: None,
                    predictions: None,
                });
            }
            MetricState::Numeric { name, entry, value } => {
                self.send(TuiMessage::Metric {
                    split,
                    name,
                    formatted: entry.formatted.clone(),
                    numeric: Some(value.current()),
                    predictions: None,
                });
            }
            MetricState::Predictions(metric) => {
                self.send(TuiMessage::Metric {
                    split,
                    name: metric.name,
                    formatted: metric.formatted,
                    numeric: None,
                    predictions: Some(metric.predictions),
                });
            }
        }
    }
}

impl MetricsRendererTraining for OxiTuiRenderer {
    fn update_train(&mut self, state: MetricState) {
        self.forward_metric(MetricSplit::Train, state);
    }

    fn update_valid(&mut self, state: MetricState) {
        self.forward_metric(MetricSplit::Valid, state);
    }

    fn render_train(&mut self, item: TrainingProgress) {
        self.send(TuiMessage::Progress {
            iteration: item.iteration,
            items_processed: item.progress.items_processed,
            items_total: item.progress.items_total,
            epoch: item.epoch,
            epoch_total: item.epoch_total,
            timestamp: Instant::now(),
        });
    }

    fn render_valid(&mut self, _item: TrainingProgress) {}

    fn on_train_end(&mut self) -> Result<(), Box<dyn std::error::Error>> {
        self.interrupter.reset();
        Ok(())
    }
}

impl MetricsRendererEvaluation for OxiTuiRenderer {
    fn update_test(&mut self, _name: EvaluationName, state: MetricState) {
        self.forward_metric(MetricSplit::Test("test".to_string()), state);
    }

    fn render_test(&mut self, _item: EvaluationProgress) {}
}

impl MetricsRenderer for OxiTuiRenderer {
    fn manual_close(&mut self) {}
}

impl Drop for OxiTuiRenderer {
    fn drop(&mut self) {
        self.send(TuiMessage::Shutdown);
        if let Some(handle) = self.handle.take() {
            let _ = handle.join();
        }
    }
}
