// Summarizes the benchmark outcomes for the scorers before the tables and plots can be generated


params.input_dir = "${launchDir}/results_sota"
params.summary_dir = "${launchDir}/summary_sota"
params.input_file = "${moduleDir}/../inputs/transf_scores.csv"

// params.input_dir = "${launchDir}/frozen_results_linear_ablations_mid"
// params.summary_dir = "${launchDir}/frozen_summary_linear_ablations_mid"
// params.input_file = "${moduleDir}/../inputs/frozen_linear_ablations_mid.csv"


params.python_path = "${moduleDir}/../"
params.conda_env = "stan"
params.tail_prob = 0.05
params.bayesian_ablation = ['HIERARCHICAL']
params.launch = false


process summarize {
    errorStrategy 'ignore'   // Uncomment for running debug workflows with partially computed correlations
    publishDir "${params.summary_dir}"
    input:
        tuple val(input_basename), val(summary_type)
    output:
        path("${out_file}"), emit: out_json
        path("${diag_file}"), optional: true, emit: diag_json
    shell:
        in_file   = "${params.input_dir}/${input_basename}"
        out_file  = "summ_${input_basename}"
        diag_file = "diag_${input_basename}"
        if (summary_type == 'correlations') {
            summary_module = "analysis.correlations_summarize"
        } else {
            error("Unknown summary type: ${summary_type}")
        }
        '''
        eval "$(conda shell.bash hook)"
        conda activate !{params.conda_env}
        export PYTHONPATH="${PYTHONPATH:+${PYTHONPATH}:}!{params.python_path}"

        python -m !{summary_module} \
            --hdi_tail_prob !{params.tail_prob} \
            --output_file '!{out_file}' \
            --input_file '!{in_file}'

        '''
}


workflow {
    // Prepare channels with the result filenames

    Channel.fromPath(params.input_file)
        .splitCsv(header:true)
        .map { row -> row.transf_metric }
        .unique()
        // .take(1) // Comment this like to run the whole workflow
        .set { all_scorers }

    Channel.fromList(params.bayesian_ablation)
        .map { directive -> [directive, directive.substring(0, 4).toLowerCase()] }
        .set { all_ablations }

    all_scorers |
        map { scorer -> tuple("correlations_${scorer}.json", 'correlations') } |
        set { correlations_results }

    // all_scorers |
    //     map { scorer -> tuple("regret_${scorer}.json", 'regret') } |
    //     set { regret_results }

    // Perform summarizations
    // correlations_results.mix(regret_results) |
    correlations_results |
        summarize
}
