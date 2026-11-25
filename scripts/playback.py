"""Shared dynamic commands and plot writer; no Isaac, torch or MuJoCo imports."""
import csv
import math
import os


def torque_differences(torque, limit, previous, before_previous, backend):
    """Normalize policy-tick differences by real actuator limits in both engines."""
    limit = backend.maximum(limit, backend.zeros_like(limit) + 1e-6)
    first = backend.zeros_like(torque) if previous is None else (torque-previous)/limit
    second = (backend.zeros_like(torque) if previous is None or before_previous is None else
              (torque-2*previous+before_previous)/limit)
    return first, second


def script_phases(
    duration: float,
    height_center: float,
    height_amp: float,
    vel_max: float,
    yaw_max: float,
    coupled_duration: float | None = None,
):
    """Build 15 phases covering three body heights and both arm extremes.

    The returned front-arm ratio is 0 at the configured lower angle (fully
    raised) and 1 at the height-dependent safe upper angle (furthest down).
    The final coupled phase defaults to twice ``duration`` while retaining the
    basic sine period, exercising two full coupled cycles by default.

    The first phase uses a cosine arm cycle that starts raised, reaches the
    lowest angle halfway through, and finishes raised.  The explicit down/up
    phase boundaries later in the script intentionally step the arm command.

    Returns ``(name, phase_duration, callable(t) -> (vx, vy, wz, height, arm_ratio))``.
    """
    coupled_duration = 2.0 * duration if coupled_duration is None else coupled_duration
    if duration <= 0.0 or coupled_duration <= 0.0:
        raise ValueError("Script phase durations must be positive.")
    omega = 2.0 * math.pi / duration

    def sin1(t: float) -> float:
        return math.sin(omega * t)

    def sin2(t: float) -> float:
        return math.sin(2.0 * omega * t)

    def height(t: float) -> float:
        return height_center + height_amp * sin1(t)

    def front_arm_cycle(t: float) -> float:
        # 0 -> 1 -> 0: raised -> furthest down -> raised, one smooth cycle.
        return 0.5 - 0.5 * math.cos(omega * t)

    phases = [
        # (1) Keep the body still at centre height and exercise only the arm.
        ("1_front_arm_cycle", duration, lambda t: (0.0, 0.0, 0.0, height_center, front_arm_cycle(t))),
        # (2) Hold the arm at the fixed fully-raised lower angle while sweeping height.
        ("2_height_sweep_arm_up", duration, lambda t: (0.0, 0.0, 0.0, height(t), 0.0)),
    ]
    heights = (
        ("mid", height_center),
        ("high", height_center + height_amp),
        ("low", height_center - height_amp),
    )
    for stage, motion, vx_peak, wz_peak in (
        (3, "walk", vel_max, 0.0),
        (4, "spin", 0.0, yaw_max),
    ):
        for height_index, (height_name, fixed_height) in enumerate(heights):
            for arm_index, (arm_name, arm_ratio) in enumerate((("down", 1.0), ("up", 0.0))):
                suffix = chr(ord("a") + 2 * height_index + arm_index)
                # Bind each loop value now so later phases retain their own
                # height, arm state and motion amplitudes.
                profile = lambda t, vx=vx_peak, wz=wz_peak, h=fixed_height, arm=arm_ratio: (
                    vx * sin1(t), 0.0, wz * sin1(t), h, arm
                )
                phases.append((f"{stage}{suffix}_{motion}_{height_name}_arm_{arm_name}", duration, profile))
    # (5) Continue coupled motion for longer; yaw has twice the base frequency.
    phases.append(
        (
            "5_coupled_arm_up",
            coupled_duration,
            lambda t: (vel_max * sin1(t), 0.0, yaw_max * sin2(t), height(t), 0.0),
        )
    )
    return phases



class TimeSeriesPlots:
    """Backends supply rows and joint/action names; plots retain Play schemas."""

    def _phase_boundaries(self) -> list[tuple[float, str]]:
        boundaries = []
        previous = None
        for row in self.rows:
            phase = str(row["phase"])
            if phase != previous:
                boundaries.append((float(row["time_s"]), phase))
                previous = phase
        return boundaries


    def save(self, output_dir: str) -> None:
        """Write one CSV plus command/state, joint and action figures."""
        if not self.rows:
            print("[WARN] No play time-series samples were collected; skipping plots.")
            return
        os.makedirs(output_dir, exist_ok=True)
        csv_path = os.path.join(output_dir, "tracking_data.csv")
        with open(csv_path, "w", newline="", encoding="utf-8") as csv_file:
            writer = csv.DictWriter(csv_file, fieldnames=list(self.rows[0].keys()))
            writer.writeheader()
            writer.writerows(self.rows)

        # Import only when exporting so ordinary play startup does not depend on
        # an interactive display.  Agg works in both GUI and --headless modes.
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        time_s = [float(row["time_s"]) for row in self.rows]
        boundaries = self._phase_boundaries()

        def decorate(axes) -> None:
            axes_list = axes.reshape(-1) if hasattr(axes, "reshape") else [axes]
            for axis in axes_list:
                axis.grid(True, alpha=0.3)
                for boundary_time, _ in boundaries[1:]:
                    axis.axvline(boundary_time, color="0.5", linewidth=0.7, alpha=0.5)
            axes_list[-1].set_xlabel("Time (s)")

        def values(key: str) -> list[float]:
            return [float(row[key]) for row in self.rows]

        tracking_panels = (
            ("command_vx_m_s", "actual_vx_m_s", "Forward velocity", "m/s"),
            ("command_vy_m_s", "actual_vy_m_s", "Lateral velocity", "m/s"),
            ("command_wz_rad_s", "actual_wz_rad_s", "Yaw rate", "rad/s"),
            ("command_height_m", "actual_height_m", "Body height", "m"),
            ("command_front_arm_deg", "actual_front_arm_deg", "Front arm angle", "deg"),
        )
        fig, axes = plt.subplots(len(tracking_panels), 1, figsize=(13, 14), sharex=True)
        for axis, (command_key, actual_key, title, unit) in zip(axes, tracking_panels):
            axis.plot(time_s, values(command_key), label="command", linewidth=1.3)
            if command_key == "command_height_m":
                axis.plot(
                    time_s,
                    values("command_height_goal_m"),
                    label="goal",
                    linewidth=1.0,
                    linestyle="--",
                )
            axis.plot(time_s, values(actual_key), label="actual", linewidth=1.0)
            axis.set_ylabel(unit)
            axis.set_title(title)
            axis.legend(loc="upper right")
        decorate(axes)
        fig.tight_layout()
        fig.savefig(os.path.join(output_dir, "body_command_tracking.png"), dpi=160)
        plt.close(fig)

        fig, axes = plt.subplots(2, 1, figsize=(13, 7), sharex=True)
        axes[0].plot(time_s, values("roll_deg"), label="roll", color="tab:blue")
        axes[1].plot(time_s, values("pitch_deg"), label="pitch", color="tab:orange")
        for axis, title in zip(axes, ("Body roll", "Body pitch")):
            axis.set_ylabel("deg")
            axis.set_title(title)
            axis.legend(loc="upper right")
        decorate(axes)
        fig.tight_layout()
        fig.savefig(os.path.join(output_dir, "body_roll_pitch.png"), dpi=160)
        plt.close(fig)

        fig, axes = plt.subplots(3, 1, figsize=(13, 9), sharex=True)
        axes[0].plot(time_s, values("front_wheel_contact_force_n"), color="tab:red")
        axes[0].set_ylabel("N")
        axes[0].set_title("Front-wheel contact force")
        axes[1].plot(time_s, values("front_wheel_clearance_m"), color="tab:green")
        axes[1].axhline(0.0, color="0.25", linewidth=0.8, linestyle="--")
        axes[1].set_ylabel("m")
        axes[1].set_title("Front-wheel ground clearance")
        axes[2].plot(time_s, values("front_arm_position_error_deg"), color="tab:purple")
        axes[2].axhline(0.0, color="0.25", linewidth=0.8, linestyle="--")
        axes[2].set_ylabel("deg")
        axes[2].set_title("Front-arm position error (actual - command)")
        decorate(axes)
        fig.tight_layout()
        fig.savefig(os.path.join(output_dir, "front_wheel_arm_diagnostics.png"), dpi=160)
        plt.close(fig)

        def save_channel_plot(
            prefix: str,
            names: list[str],
            title: str,
            unit: str,
            filename: str,
            columns: int = 1,
        ) -> None:
            columns = max(1, min(columns, len(names)))
            rows = math.ceil(len(names) / columns)
            fig, axes = plt.subplots(
                rows,
                columns,
                figsize=(7.0 * columns, 2.5 * rows),
                sharex=True,
                squeeze=False,
            )
            flat_axes = axes.reshape(-1)
            for axis, name in zip(flat_axes, names):
                axis.plot(time_s, values(f"{prefix}/{name}"), linewidth=0.9)
                axis.set_ylabel(unit)
                axis.set_title(name)
            for axis in flat_axes[len(names) :]:
                axis.set_visible(False)
            decorate(axes)
            fig.suptitle(title)
            fig.tight_layout(rect=(0.0, 0.0, 1.0, 0.98))
            fig.savefig(os.path.join(output_dir, filename), dpi=160)
            plt.close(fig)

        save_channel_plot(
            "joint_position_rad", self.joint_names, "All joint positions", "rad",
            "all_joint_position.png", columns=3,
        )
        save_channel_plot(
            "joint_velocity_rad_s", self.joint_names, "All joint velocities", "rad/s",
            "all_joint_velocity.png", columns=3,
        )
        save_channel_plot(
            "joint_acceleration_rad_s2", self.joint_names, "All joint accelerations", "rad/s²",
            "all_joint_acceleration.png", columns=3,
        )
        save_channel_plot(
            "policy_action", self.action_names, "Raw policy actions", "normalized",
            "policy_actions.png", columns=2,
        )
        save_channel_plot(
            "applied_torque_nm", self.actuated_joint_names, "Applied actuator torques", "N·m",
            "applied_torque.png", columns=2,
        )
        save_channel_plot(
            "normalized_torque_delta", self.actuated_joint_names,
            "Applied torque first differences (normalized by effort limit)", "normalized",
            "normalized_torque_delta.png", columns=2,
        )
        save_channel_plot(
            "normalized_torque_second_diff", self.actuated_joint_names,
            "Applied torque second differences (normalized by effort limit)", "normalized",
            "normalized_torque_second_diff.png", columns=2,
        )

        phase_path = os.path.join(output_dir, "phases.csv")
        with open(phase_path, "w", newline="", encoding="utf-8") as phase_file:
            writer = csv.writer(phase_file)
            writer.writerow(("start_time_s", "phase"))
            writer.writerows(boundaries)
        print(f"[INFO] Saved play time-series CSV and plots to: {output_dir}")


