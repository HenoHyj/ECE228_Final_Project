# Implementation Plan: Continuous-Time Neural ODEs for Coastal Dynamics

## Phase 1: Data Acquisition (NOAA API Pipeline)
The NOAA CO-OPS API has strict limitations (e.g., maximum 31-day data pull per request for 6-minute interval data). To fetch multi-year data, we need a robust programmatic pipeline.

### 1.1 API Endpoint & Parameters
* **Base URL:** `https://api.tidesandcurrents.noaa.gov/api/prod/datagetter`
* **Station ID:** `9410230` (La Jolla, Scripps Pier)
* **Target Products:**
    * `water_level` (Datum: MLLW)
    * `air_temperature`
    * `water_temperature`
    * `barometric_pressure`
    * `wind` (speed and direction)
* **Format:** `json` or `csv`
* **Timezone:** `gmt` (Standardize everything to GMT to avoid Daylight Saving Time gaps).

### 1.2 Fetching Strategy
1.  **Date Generation:** Use `pandas.date_range` to generate monthly start and end dates spanning the last 2-3 years (e.g., Jan 2023 to Present).
2.  **Sequential Requests:** Loop through the generated date chunks. For each month, make separate requests for each of the target products (since the API often restricts fetching multiple different physical products in a single call).
3.  **Error Handling & Rate Limiting:** Implement `try-except` blocks and `time.sleep()` to handle API rate limits and connection timeouts gracefully.
4.  **Merge & Align:** Merge the responses on the `timestamp` column using an outer join (`pd.merge(how='outer')`).

## Phase 2: Data Preprocessing & Dataloaders
Since our core methodology explicitly tests irregular sampling, preprocessing must be handled carefully.

1.  **Do NOT Impute Everything:** Keep the `NaN` values where sensors dropped out. 
2.  **Time Normalization:** Convert timestamps into a continuous float variable $t$ (e.g., hours or days since the start of the sequence). Neural ODEs require $t$ as an explicit input.
3.  **Feature Normalization:** Standardize all physical variables to zero mean and unit variance (`StandardScaler`).
4.  **Sequence Splitting:** Create sliding windows (e.g., use the past 24 hours to predict the next 12 hours). 
5.  **Mask Generation:** Create a boolean mask indicating which time steps are valid (observed) and which are `NaN`. This mask is crucial for calculating the Loss function only on valid observations.

## Phase 3: Baseline Implementation (LSTM)
Set up the discrete-time baseline to prove the necessity of the Neural ODE.

1.  **Architecture:** A standard PyTorch `nn.LSTM` or `nn.GRU`.
2.  **Handling NaNs for Baseline:** Since standard LSTMs cannot ingest `NaN` values, we must apply a basic imputation strategy (e.g., Forward-Fill or Linear Interpolation) *only* for the baseline model's input.
3.  **Training:** Train to minimize MSE over the prediction horizon.

## Phase 4: Neural ODE Implementation
This is the core SciML component. 

1.  **Environment:** Install `torchdiffeq` (`pip install torchdiffeq`).
2.  **ODEFunc (The Derivative Network):** * Define a PyTorch `nn.Module` that outputs $dh/dt$. 
    * Input: `(t, h)`. Output: $dh/dt$ (same dimension as $h$).
    * Keep it relatively simple: 2-3 linear layers with $\tanh$ or $\text{ELU}$ activations.
3.  **ODEBlock (The Integrator):**
    * Wrap the `ODEFunc` using `torchdiffeq.odeint_adjoint`.
    * Pass the initial state $h(t_0)$ and the specific, potentially irregular time points $t_1, t_2, \dots, t_N$ where we have observations.
4.  **Encoder/Decoder Architecture (Latent ODE):**
    * **Encoder:** An RNN that processes the history backwards to generate the initial latent state $h(t_0)$.
    * **ODE:** Integrates $h(t_0)$ forward in time to $h(t_i)$.
    * **Decoder:** A linear layer that maps the latent state $h(t_i)$ back to the physical variables (temperature, pressure, etc.).

## Phase 5: Training & Robustness Evaluation
1.  **Standard Training:** Train both models on the training set using standard MSE loss on the predicted horizon.
2.  **Robustness Test (The Selling Point):** * Take a clean test set.
    * Artificially drop $20\%$, $40\%$, and $60\%$ of the data points in the input sequences to simulate severe sensor failure.
    * Compare the performance drop between the Baseline (which relies on imputation) and the Neural ODE (which naturally integrates over the gaps).